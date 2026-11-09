import { h, fmtNum, fmtBytes } from "./ui";

type Layer = { index: number; kind: string; attention: number; mlp: number; norm: number };
type Anatomy = {
  error?: string;
  quant: string;
  bits: number;
  path: string;
  quants: string[];
  diskBytes: number;
  tensorCount: number;
  total: number;
  groups: Record<string, number>;
  layers: Layer[];
  config: {
    architecture: string;
    hiddenSize: number;
    intermediateSize: number;
    numLayers: number;
    numHeads: number;
    numKvHeads: number;
    headDim: number;
    vocabSize: number;
    maxContext: number;
    fullAttentionInterval: number;
    activation: string;
    license: string | null;
  };
};

const COLORS = {
  embedding: "#8a6fb0",
  attention: "#5d7f99",
  mlp: "#c9a25c",
  lm_head: "#b07a6f",
  vision: "#5f8f78",
  other: "#555",
};

export function mountAnatomy(root: HTMLElement) {
  let loaded = "";
  const body = h("div", { class: "grid" }, h("p", { class: "dim" }, "Reading the weight files…"));
  root.append(body);

  async function load(quant = "") {
    const r = await fetch("/api/anatomy" + (quant ? `?quant=${encodeURIComponent(quant)}` : ""));
    const a: Anatomy = await r.json();
    body.replaceChildren(a.error ? h("div", { class: "card appear" }, h("p", {}, a.error)) : render(a));
    loaded = a.quant ?? "";
  }

  function render(a: Anatomy) {
    const c = a.config;
    const attn = a.layers.reduce((s, l) => s + l.attention + l.norm, 0);
    const mlp = a.layers.reduce((s, l) => s + l.mlp, 0);
    const language = a.total - (a.groups.vision ?? 0);
    const neuronsPerLayer = c.intermediateSize;
    const neurons = neuronsPerLayer * c.numLayers;
    const full = a.layers.filter((l) => l.kind === "full_attention").length;

    const parts: [keyof typeof COLORS, string, number][] = [
      ["embedding", "word embeddings (input)", a.groups.embedding ?? 0],
      ["attention", "attention (how words look at each other)", attn],
      ["mlp", "MLP / feed-forward (the “neurons”, most of the knowledge)", mlp],
      ["lm_head", "output head (picks the next word)", a.groups.lm_head ?? 0],
      ["vision", "vision tower (reads images)", a.groups.vision ?? 0],
    ];

    const stat = (v: string, k: string) => h("div", { class: "stat" }, h("div", { class: "v" }, v), h("div", { class: "k" }, k));
    const quantSel = h("select", { style: "width:auto", onchange: () => load(quantSel.value) },
      ...a.quants.map((q) => h("option", { value: q, selected: q === a.quant }, q)));

    const layerTip = (l: Layer) =>
      `layer ${l.index} · ${l.kind === "full_attention" ? "full attention (keeps a KV cache)" : "linear attention (fixed-size memory)"}\n` +
      `attention ${fmtNum(l.attention)} · MLP ${fmtNum(l.mlp)} params`;

    return h("div", { class: "grid appear" },
      h("div", { class: "card" },
        h("div", { class: "row", style: "justify-content:space-between" },
          h("h2", {}, "How big is it?"),
          h("div", { class: "row dim", style: "font-size:12.5px" }, "build", quantSel)),
        h("p", { class: "dim" }, "Counted from the safetensors headers on disk, not from the model card. Nothing is loaded into memory."),
        h("div", { class: "stats" },
          stat(fmtNum(language), "parameters in the language model"),
          stat(fmtNum(a.total), "with the vision tower"),
          stat(fmtBytes(a.diskBytes), `on disk (${a.bits} bits per weight)`),
          stat(fmtBytes((language * 2)), "the same model at full 16-bit"),
          stat(String(c.numLayers), `layers (${full} full attention, ${c.numLayers - full} linear)`),
          stat(fmtNum(neurons), `MLP neurons (${neuronsPerLayer.toLocaleString()} per layer)`),
          stat(c.hiddenSize.toLocaleString(), "numbers per word inside the model"),
          stat(fmtNum(c.vocabSize), "tokens in the vocabulary"),
          stat(fmtNum(c.maxContext), "max context it was trained for"),
          stat(a.tensorCount.toLocaleString(), "tensors in the files"),
        ),
      ),

      h("div", { class: "card" },
        h("h2", {}, "Where the parameters are"),
        h("div", { class: "stack-bar" }, ...parts.map(([k, , n]) => {
          const d = h("div", { title: `${k}: ${fmtNum(n)}` });
          d.style.background = COLORS[k];
          d.style.flexGrow = String(n);
          return d;
        })),
        h("div", { class: "keyrow" }, ...parts.map(([k, label, n]) => {
          const i = h("i");
          i.style.background = COLORS[k];
          return h("span", {}, i, `${label} · ${fmtNum(n)} (${((n / a.total) * 100).toFixed(0)}%)`);
        })),
      ),

      h("div", { class: "card" },
        h("h2", {}, `The ${c.numLayers} layers, in order`),
        h("p", { class: "dim" },
          `A word passes through all ${c.numLayers} layers, top-left to bottom-right, before the next word is picked. ` +
          `Every ${c.fullAttentionInterval}th layer (brass) is full attention and keeps a memory of every previous token; ` +
          `the rest (blue) use linear attention with a fixed-size memory. That is why long context is cheap in RAM here.`),
        h("div", { class: "layers" }, ...a.layers.map((l) => h("div", { class: `layer ${l.kind}`, title: layerTip(l) }, String(l.index)))),
      ),

      h("div", { class: "card prose" },
        h("h2", {}, "What a “neuron” is in this model"),
        h("p", {}, `Each layer has an MLP block with ${neuronsPerLayer.toLocaleString()} neurons. A neuron reads the ${c.hiddenSize.toLocaleString()}-number vector for the current word, ` +
          `computes one number (${c.activation} activation, gated), and writes a ${c.hiddenSize.toLocaleString()}-number contribution back. ` +
          `That is 3 × ${c.hiddenSize.toLocaleString()} = ${(3 * c.hiddenSize).toLocaleString()} weights per neuron, and ${neurons.toLocaleString()} neurons in all: ` +
          `${fmtNum(mlp)} parameters, ${((mlp / language) * 100).toFixed(0)}% of the language model.`),
        h("p", {}, "The parameters are the weights on the connections, not the neurons. When people say “27B” they mean ~27 billion of those weights."),
        h("p", {}, "Individual neurons rarely mean one clean thing. Most concepts are spread across thousands of neurons, and each neuron takes part in many concepts (“superposition”). That is why you can't make the model smarter by turning single neurons up or down. See the Tuning tab."),
      ),

      h("div", { class: "card prose" },
        h("h2", {}, "Is it open?"),
        h("p", { class: "bigq" }, "Open weights: yes. Open source: no."),
        h("p", {}, `Every one of those ${fmtNum(language)} numbers is in ${a.path} on your disk, and you can read, change, fine-tune and redistribute them` +
          ` under the license (${c.license ?? "Apache-2.0, per the fine-tune's model card"}).`),
        h("p", {}, "What you don't get is how it was made: the training data, the training code and the compute. Alibaba trained the base Qwen; a third party then fine-tuned it to stop refusing. You can change the result, but you can't rebuild it from scratch."),
        h("p", { class: "dim" }, `Architecture: ${c.architecture} · ${c.numHeads} query heads, ${c.numKvHeads} KV heads, head size ${c.headDim}.`),
      ),

      h("div", { class: "card" },
        h("h2", {}, "Per layer"),
        h("table", {},
          h("tr", {}, h("th", {}, "layer"), h("th", {}, "type"), h("th", { class: "num" }, "attention"), h("th", { class: "num" }, "MLP"), h("th", { class: "num" }, "total")),
          ...a.layers.map((l) => h("tr", {},
            h("td", {}, String(l.index)),
            h("td", { class: "dim" }, l.kind.replace("_", " ")),
            h("td", { class: "num" }, fmtNum(l.attention)),
            h("td", { class: "num" }, fmtNum(l.mlp)),
            h("td", { class: "num" }, fmtNum(l.attention + l.mlp + l.norm)))),
        ),
      ),
    );
  }

  return { show: () => { if (!loaded) load(); } };
}
