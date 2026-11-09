import "./style.css";
import { health, serverCommand, type Health } from "./api";
import { mountChat } from "./chat";
import { mountAnatomy } from "./anatomy";
import { mountTests } from "./tests";
import { mountTuning } from "./tuning";
import { swapText } from "./ui";

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

// Tabs: a sliding highlight follows the selected tab; views cross-fade.
const tabs = $<HTMLElement>("#tabs");
const mark = $<HTMLElement>("#tab-mark");
function select(name: string) {
  for (const b of tabs.querySelectorAll("button")) b.classList.toggle("on", b.dataset.tab === name);
  for (const v of document.querySelectorAll(".view")) v.classList.toggle("on", v.id === name);
  const on = tabs.querySelector<HTMLElement>("button.on")!;
  mark.style.left = on.offsetLeft + "px";
  mark.style.width = on.offsetWidth + "px";
  try { localStorage.setItem("tab", name); } catch {}
  if (name === "anatomy") anatomy.show();
}
tabs.addEventListener("click", (e) => {
  const b = (e.target as HTMLElement).closest("button");
  if (b?.dataset.tab) select(b.dataset.tab);
});

// Server status. Polled gently (and not at all while the tab is hidden) so the GUI never
// keeps the model loaded: the server unloads itself after a few idle minutes.
const dot = $<HTMLElement>("#status-dot");
const text = $<HTMLElement>("#status-text");
const loadBtn = $<HTMLButtonElement>("#load-btn");
let state: Health = { up: false };
let busy = false;
let checked = false;
const listeners: ((h: Health) => void)[] = [];
export const onHealth = (f: (h: Health) => void) => { listeners.push(f); f(state); };

function render() {
  dot.className = "dot " + (busy ? "loading" : state.up ? "up" : "");
  const model = state.model?.split("/").slice(-2).join("/") ?? "";
  if (!checked && !busy) return;
  swapText(text, busy ? "working…" : state.up ? `Qwen loaded · ${model}` : "model not loaded");
  swapText(loadBtn, state.up ? "Unload" : "Load model");
  loadBtn.disabled = busy;
}

async function poll() {
  if (document.hidden && !busy) return;
  const h = await health();
  if (!checked || h.up !== state.up || h.model !== state.model) {
    checked = true;
    state = h;
    listeners.forEach((f) => f(h));
    render();
  }
}

loadBtn.addEventListener("click", async () => {
  busy = true;
  render();
  try {
    const r = await serverCommand(state.up ? "down" : "up");
    if (!r.ok) alert(r.output || "porthole command failed");
  } finally {
    busy = false;
    await poll();
    render();
  }
});

const chat = mountChat($("#chat"), () => state);
const anatomy = mountAnatomy($("#anatomy"));
mountTests($("#tests"), () => state);
mountTuning($("#tuning"));
void chat;

let saved = "chat";
try { saved = localStorage.getItem("tab") || "chat"; } catch {}
requestAnimationFrame(() => select(saved));
window.addEventListener("resize", () => select(tabs.querySelector<HTMLElement>("button.on")!.dataset.tab!));
render();
poll();
setInterval(poll, 15000);
document.addEventListener("visibilitychange", poll);
