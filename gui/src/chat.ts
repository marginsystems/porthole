import { chat, type Health, type Msg, type Sampling, type TokenInfo } from "./api";
import { h, swapText, confidenceColor, pct, showToken } from "./ui";

type Turn = { role: "user" | "assistant"; content: string };

const DEFAULTS: Sampling & { system: string; confidence: boolean } = {
  system: "",
  temperature: 0.7,
  top_p: 0.95,
  top_k: 20,
  max_tokens: 1024,
  enable_thinking: false,
  confidence: true,
};

function loadSettings(): typeof DEFAULTS {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("chat-settings") || "{}") };
  } catch {
    return { ...DEFAULTS };
  }
}

export function mountChat(root: HTMLElement, server: () => Health) {
  const s = loadSettings();
  const save = () => { try { localStorage.setItem("chat-settings", JSON.stringify(s)); } catch {} };
  const history: Turn[] = [];
  let abort: AbortController | null = null;

  // ---- side panel: the knobs you can turn without changing a single weight
  const knob = (key: "temperature" | "top_p" | "top_k" | "max_tokens", label: string, min: number, max: number, step: number, hint: string) => {
    const val = h("b", {}, String(s[key]));
    const input = h("input", {
      type: "range", min, max, step, value: String(s[key]),
      oninput: () => { (s as any)[key] = Number(input.value); val.textContent = input.value; save(); },
    });
    return h("div", { class: "knob" }, h("label", {}, label, val), input, h("div", { class: "hint" }, hint));
  };
  const check = (key: "enable_thinking" | "confidence", label: string, hint: string) => {
    const input = h("input", { type: "checkbox", checked: s[key], onchange: () => { s[key] = input.checked; save(); } });
    return h("div", { class: "knob" }, h("label", { style: "justify-content:flex-start;gap:8px;color:var(--text)" }, input, label), h("div", { class: "hint" }, hint));
  };
  const system = h("textarea", {
    rows: 4,
    placeholder: "System prompt (optional). E.g. “You are a blunt code reviewer.”",
    value: s.system,
    oninput: () => { s.system = system.value; save(); },
  });

  const side = h("aside", { class: "side" },
    h("div", { class: "card grid" },
      h("h2", {}, "Sampling"),
      knob("temperature", "temperature", 0, 2, 0.05, "Randomness. 0 = always the likeliest word; >1 gets weird."),
      knob("top_p", "top_p", 0.05, 1, 0.05, "Only sample from the words that make up this much probability."),
      knob("top_k", "top_k", 1, 100, 1, "Only sample from the k likeliest words."),
      knob("max_tokens", "max_tokens", 64, 4096, 64, "Longest answer, in tokens (~¾ of a word each)."),
      check("enable_thinking", "thinking", "Let Qwen reason privately before answering. Slower, often better on hard problems."),
      check("confidence", "colour by confidence", "Shade each token by how sure the model was. Hover for the runners-up."),
    ),
    h("div", { class: "card grid" }, h("h2", {}, "System prompt"), system),
    h("button", { class: "ghost", onclick: () => { abort?.abort(); history.length = 0; log.replaceChildren(empty()); } }, "Clear conversation"),
  );

  // ---- conversation
  const empty = () => h("div", { class: "empty appear" },
    h("p", {}, "Talk to the local Qwen directly: no harness, no tools, just the model."),
    h("p", {}, "Turn on “colour by confidence” and watch where it hesitates: that is where it is guessing."));
  const log = h("div", { class: "log" }, empty());
  const input = h("textarea", { rows: 2, placeholder: "Message Qwen…  (Enter sends, Shift+Enter is a new line)" });
  const sendBtn = h("button", { class: "primary" }, "Send");
  const composer = h("div", { class: "composer" }, input, sendBtn);
  root.append(h("div", { class: "chat-wrap" }, h("div", { class: "chat-main" }, log, composer), side));

  const tip = h("div", { class: "tip" });
  document.body.append(tip);

  function tokenSpan(t: TokenInfo) {
    const span = h("span", { class: "tok" }, t.token);
    span.style.background = confidenceColor(t.logprob);
    span.addEventListener("mouseenter", () => {
      const rows = [`${showToken(t.token).padEnd(14)} ${pct(t.logprob)}  ← picked`];
      for (const alt of t.top) if (alt.token !== t.token) rows.push(`${showToken(alt.token).padEnd(14)} ${pct(alt.logprob)}`);
      if (!t.top.length) rows.push("(start the server with TOP_LOGPROBS_K=5 to see runners-up)");
      tip.textContent = rows.join("\n");
      const r = span.getBoundingClientRect();
      tip.style.left = Math.min(r.left, innerWidth - 320) + "px";
      tip.style.top = r.bottom + 6 + "px";
      tip.classList.add("on");
    });
    span.addEventListener("mouseleave", () => tip.classList.remove("on"));
    return span;
  }

  async function send() {
    const text = input.value.trim();
    if (!text || abort) return;
    if (!server().up) {
      alert("The model isn't loaded. Click “Load model” at the top right (takes 20–60 s).");
      return;
    }
    log.querySelector(".empty")?.remove();
    input.value = "";
    history.push({ role: "user", content: text });
    log.append(h("div", { class: "msg user appear" }, text));

    const think = h("div", { class: "think" });
    const thinkBox = h("div", { class: "collapsible" }, h("div", {}, think));
    const body = h("div", {});
    const meta = h("div", { class: "meta" }, "…");
    const bubble = h("div", { class: "msg assistant appear" }, thinkBox, body, meta);
    log.append(bubble);
    log.scrollTop = log.scrollHeight;

    const messages: Msg[] = [...(s.system ? [{ role: "system" as const, content: s.system }] : []), ...history];
    abort = new AbortController();
    swapText(sendBtn, "Stop");
    const useTokens = s.confidence;
    try {
      const { temperature, top_p, top_k, max_tokens, enable_thinking } = s;
      const r = await chat(messages, { temperature, top_p, top_k, max_tokens, enable_thinking }, {
        onReasoning: (t) => { think.append(t); thinkBox.classList.add("open"); },
        onContent: (t) => { if (!useTokens) body.append(t); },
        onToken: (t) => { if (useTokens && t.part === "content") body.append(tokenSpan(t)); },
      }, abort.signal);
      // Tokens can be absent (older server); fall back to plain text.
      if (useTokens && !body.childNodes.length) body.textContent = r.content;
      history.push({ role: "assistant", content: r.content });
      const answer = r.tokens.filter((t) => t.part === "content");
      const avg = answer.length ? answer.reduce((a, t) => a + Math.exp(t.logprob), 0) / answer.length : NaN;
      const unsure = answer.filter((t) => Math.exp(t.logprob) < 0.5).length;
      const n = r.completionTokens ?? r.tokens.length;
      const gen = r.seconds - (r.firstTokenSeconds ?? 0);
      swapText(meta, [
        `${n} tokens`,
        `${(n / Math.max(gen, 0.01)).toFixed(1)} tok/s`,
        r.firstTokenSeconds !== undefined ? `first token ${r.firstTokenSeconds.toFixed(1)} s` : "",
        r.promptTokens ? `prompt ${r.promptTokens} tok` : "",
        isNaN(avg) ? "" : `avg confidence ${(avg * 100).toFixed(0)}%`,
        isNaN(avg) ? "" : `${unsure} unsure tokens`,
      ].filter(Boolean).join("  ·  "));
    } catch (e: any) {
      history.pop();
      swapText(meta, e?.name === "AbortError" ? "stopped (not kept in the conversation)" : `error: ${e?.message ?? e}`);
    } finally {
      abort = null;
      swapText(sendBtn, "Send");
      log.scrollTop = log.scrollHeight;
    }
  }

  sendBtn.addEventListener("click", () => (abort ? abort.abort() : send()));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  return {};
}
