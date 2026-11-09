import { defineConfig, type Plugin } from "vite";
import { Agent } from "node:http";
import { spawn } from "node:child_process";
import { existsSync, openSync, readSync, closeSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// The Qwen server porthole runs (start-server.sh). Chat requests go through the dev server's proxy
// so the browser never needs CORS.
const QWEN_URL = process.env.QWEN_URL || "http://127.0.0.1:8080";
const WEIGHTS_NAME = "Qwen3.8-27B-Uncensored-MLX";
const here = dirname(fileURLToPath(import.meta.url));

// The weights live next to the porthole script. From a git worktree they are in the main checkout,
// a few folders up, so walk up until we find them. PORTHOLE_WEIGHTS overrides.
function weightsRoot(): string | null {
  if (process.env.PORTHOLE_WEIGHTS) return resolve(process.env.PORTHOLE_WEIGHTS);
  let dir = resolve(here, "..");
  for (let i = 0; i < 8; i++) {
    const cand = join(dir, WEIGHTS_NAME);
    if (existsSync(cand)) return cand;
    const up = dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  return null;
}

type Tensor = { dtype: string; shape: number[] };

// A safetensors file starts with an 8-byte little-endian length, then a JSON header listing every
// tensor's dtype and shape. Reading only that header gives exact sizes without loading 16 GB.
function readHeader(file: string): Record<string, Tensor> {
  const fd = openSync(file, "r");
  try {
    const lenBuf = Buffer.alloc(8);
    readSync(fd, lenBuf, 0, 8, 0);
    const len = Number(lenBuf.readBigUInt64LE(0));
    if (len > 100 * 2 ** 20) throw new Error(`header of ${file} is implausibly large`);
    const buf = Buffer.alloc(len);
    readSync(fd, buf, 0, len, 8);
    const json = JSON.parse(buf.toString("utf8"));
    delete json.__metadata__;
    return json;
  } finally {
    closeSync(fd);
  }
}

const prod = (s: number[]) => s.reduce((a, b) => a * b, 1);

function anatomy(quant: string) {
  const root = weightsRoot();
  if (!root) return { error: `No ${WEIGHTS_NAME} folder found. Run \`porthole pull\`, or set PORTHOLE_WEIGHTS.` };
  const dir = join(root, quant);
  if (!existsSync(join(dir, "config.json"))) return { error: `${dir} has no config.json` };
  const config = JSON.parse(readFileSync(join(dir, "config.json"), "utf8"));
  const bits: number = config.quantization?.bits ?? 16;

  const files = readdirSync(dir).filter((f) => f.endsWith(".safetensors")).sort();
  const tensors: Record<string, Tensor> = {};
  let diskBytes = 0;
  for (const f of files) {
    Object.assign(tensors, readHeader(join(dir, f)));
    diskBytes += statSync(join(dir, f)).size;
  }

  // Quantized weights are packed: 32/bits values per uint32. `.scales` and `.biases` that sit
  // next to a quantized weight are bookkeeping for the packing, not parameters of the model.
  const params = (name: string, t: Tensor): number => {
    if (name.endsWith(".scales") || (name.endsWith(".biases") && tensors[name.replace(/\.biases$/, ".scales")]))
      return 0;
    if (t.dtype === "U32" && tensors[name.replace(/\.weight$/, ".scales")]) return prod(t.shape) * (32 / bits);
    return prod(t.shape);
  };

  const groups: Record<string, number> = {};
  const layers: { index: number; kind: string; attention: number; mlp: number; norm: number }[] = [];
  const add = (g: string, n: number) => (groups[g] = (groups[g] ?? 0) + n);
  let total = 0;
  for (const [name, t] of Object.entries(tensors)) {
    const n = params(name, t);
    total += n;
    const m = name.match(/^language_model\.model\.layers\.(\d+)\.(\w+)/);
    if (m) {
      const i = Number(m[1]);
      const L = (layers[i] ??= { index: i, kind: "", attention: 0, mlp: 0, norm: 0 });
      if (m[2] === "mlp") L.mlp += n;
      else if (m[2] === "self_attn") { L.attention += n; L.kind = "full_attention"; }
      else if (m[2] === "linear_attn") { L.attention += n; L.kind ||= "linear_attention"; }
      else L.norm += n;
      add("layers", n);
    } else if (name.startsWith("vision_tower")) add("vision", n);
    else if (name.includes("embed_tokens")) add("embedding", n);
    else if (name.includes("lm_head")) add("lm_head", n);
    else add("other", n);
  }

  const tc = config.text_config ?? config;
  return {
    quant,
    bits,
    path: dir,
    quants: readdirSync(root).filter((q) => existsSync(join(root, q, "config.json"))),
    diskBytes,
    tensorCount: Object.keys(tensors).length,
    total,
    groups,
    layers: layers.filter(Boolean),
    config: {
      architecture: config.architectures?.[0],
      hiddenSize: tc.hidden_size,
      intermediateSize: tc.intermediate_size,
      numLayers: tc.num_hidden_layers,
      numHeads: tc.num_attention_heads,
      numKvHeads: tc.num_key_value_heads,
      headDim: tc.head_dim,
      vocabSize: tc.vocab_size,
      maxContext: tc.max_position_embeddings,
      fullAttentionInterval: tc.full_attention_interval,
      activation: tc.hidden_act,
      license: readLicense(dir),
    },
  };
}

function readLicense(dir: string): string | null {
  try {
    const card = readFileSync(join(dir, "README.md"), "utf8");
    return card.match(/^license:\s*(.+)$/m)?.[1]?.trim() ?? null;
  } catch {
    return null;
  }
}

// `porthole up` / `porthole down` from the checkout that holds the weights (it has the venv).
// Only ever run when the user clicks Load or Unload in the GUI.
function runPorthole(cmd: "up" | "down"): Promise<{ ok: boolean; output: string }> {
  const root = weightsRoot();
  // With PORTHOLE_WEIGHTS the weights can live anywhere, so the script comes from this checkout.
  const checkout = process.env.PORTHOLE_WEIGHTS ? resolve(here, "..") : root ? dirname(root) : "";
  const script = checkout ? join(checkout, "porthole") : "";
  if (!script || !existsSync(script)) return Promise.resolve({ ok: false, output: "porthole script not found next to the weights" });
  return new Promise((done) => {
    const p = spawn(script, [cmd], { cwd: dirname(script), stdio: ["ignore", "pipe", "pipe"] });
    let output = "";
    p.stdout.on("data", (d) => (output += d));
    p.stderr.on("data", (d) => (output += d));
    p.on("error", (err) => done({ ok: false, output: String(err) }));
    p.on("close", (code) => done({ ok: code === 0, output: output.replace(/\x1b\[[0-9;]*m/g, "").trim() }));
  });
}

function portholeApi(): Plugin {
  return {
    name: "porthole-api",
    configureServer(server) {
      for (const cmd of ["up", "down"] as const) {
        server.middlewares.use(`/api/${cmd}`, async (req, res) => {
          if (req.method !== "POST") { res.statusCode = 405; return res.end(); }
          // Any page can send a bare cross-site POST; only the lab's own origin may start or stop the model.
          // Compare against the expected local origins, not the request's Host (attacker-controlled under DNS rebinding).
          const local = ["localhost", "127.0.0.1"].map((h) => `http://${h}:${server.config.server.port}`);
          if (req.headers.origin && !local.includes(req.headers.origin)) { res.statusCode = 403; return res.end(); }
          res.setHeader("content-type", "application/json");
          res.end(JSON.stringify(await runPorthole(cmd)));
        });
      }
      server.middlewares.use("/api/anatomy", (req, res) => {
        const quant = new URL(req.url ?? "", "http://x").searchParams.get("quant") || process.env.QWEN_QUANT || "4-bit";
        res.setHeader("content-type", "application/json");
        try {
          if (!/^\w[\w.-]*$/.test(quant)) throw new Error("bad quant name");
          res.end(JSON.stringify(anatomy(quant)));
        } catch (e) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: String(e) }));
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [portholeApi()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    // Fail at startup rather than fall back to 5174: the Origin allowlist in /api/up and /api/down is built from this port.
    strictPort: true,
    proxy: {
      "/qwen": {
        target: QWEN_URL,
        changeOrigin: true,
        // No keep-alive: an idle pooled socket counts as "in use" to the server's idle unloader,
        // so the model would never free its memory while the GUI is open.
        agent: new Agent({ keepAlive: false }),
        rewrite: (p) => p.replace(/^\/qwen/, ""),
      },
    },
  },
});
