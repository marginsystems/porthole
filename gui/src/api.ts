// Talks to the local Qwen server (mlx_vlm, OpenAI-compatible) through the Vite proxy at /qwen.

export type Msg = { role: "system" | "user" | "assistant"; content: string };

export type Sampling = {
  temperature: number;
  top_p: number;
  top_k: number;
  max_tokens: number;
  enable_thinking: boolean;
  seed?: number;
};

/** One generated token. `part` says whether it belongs to the hidden thinking or the answer. */
export type TokenInfo = {
  token: string;
  logprob: number;
  top: { token: string; logprob: number }[];
  part: "reasoning" | "content";
};

export type StreamEvents = {
  onContent?: (text: string) => void;
  onReasoning?: (text: string) => void;
  onToken?: (t: TokenInfo) => void;
};

export type Health = { up: boolean; model?: string; context?: number };

export async function health(): Promise<Health> {
  try {
    const r = await fetch("/qwen/health", { cache: "no-store" });
    if (!r.ok) return { up: false };
    const j = await r.json();
    return { up: true, model: j.loaded_model ?? undefined, context: j.effective_context_limit ?? undefined };
  } catch {
    return { up: false };
  }
}

export async function serverCommand(cmd: "up" | "down"): Promise<{ ok: boolean; output: string }> {
  const r = await fetch(`/api/${cmd}`, { method: "POST" });
  return r.json();
}

let modelId: string | null = null;
async function model(): Promise<string> {
  if (modelId) return modelId;
  const r = await fetch("/qwen/v1/models");
  const j = await r.json();
  modelId = j.data?.[0]?.id ?? "default";
  return modelId!;
}

export type ChatResult = {
  content: string;
  reasoning: string;
  tokens: TokenInfo[];
  promptTokens?: number;
  completionTokens?: number;
  seconds: number;
  firstTokenSeconds?: number;
};

/** Streams one chat completion. Resolves with the whole result when the stream ends. */
export async function chat(
  messages: Msg[],
  s: Sampling,
  ev: StreamEvents = {},
  signal?: AbortSignal,
  topLogprobs = 5,
): Promise<ChatResult> {
  const t0 = performance.now();
  const body = {
    model: await model(),
    messages,
    stream: true,
    stream_options: { include_usage: true },
    logprobs: true,
    top_logprobs: topLogprobs,
    ...s,
  };
  const r = await fetch("/qwen/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`server answered ${r.status}: ${await r.text()}`);

  const out: ChatResult = { content: "", reasoning: "", tokens: [], seconds: 0 };
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      let j: any;
      try {
        j = JSON.parse(data);
      } catch {
        continue;
      }
      if (j.usage) {
        out.promptTokens = j.usage.prompt_tokens;
        out.completionTokens = j.usage.completion_tokens;
      }
      const ch = j.choices?.[0];
      if (!ch) continue;
      const d = ch.delta ?? {};
      const reasoning = d.reasoning ?? d.reasoning_content;
      if (out.firstTokenSeconds === undefined && (d.content || reasoning))
        out.firstTokenSeconds = (performance.now() - t0) / 1000;
      if (reasoning) {
        out.reasoning += reasoning;
        ev.onReasoning?.(reasoning);
      }
      if (d.content) {
        out.content += d.content;
        ev.onContent?.(d.content);
      }
      for (const lp of ch.logprobs?.content ?? []) {
        const t: TokenInfo = {
          token: lp.token,
          logprob: lp.logprob,
          top: (lp.top_logprobs ?? []).map((x: any) => ({ token: x.token, logprob: x.logprob })),
          part: reasoning && !d.content ? "reasoning" : "content",
        };
        out.tokens.push(t);
        ev.onToken?.(t);
      }
    }
  }
  out.seconds = (performance.now() - t0) / 1000;
  return out;
}
