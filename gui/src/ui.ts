// Small DOM helpers shared by the views.

export const SPRING = "linear(0, 0.072, 0.249, 0.47, 0.677, 0.84, 0.953, 1.02, 1.052, 1.06, 1.054, 1.04, 1.025, 1.012, 1.003, 0.998, 0.997, 0.998, 1)";

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, any> = {},
  ...kids: (Node | string | null | undefined | false)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k in el) (el as any)[k] = v;
    else el.setAttribute(k, String(v));
  }
  for (const c of kids) if (c !== null && c !== undefined && c !== false) el.append(c);
  return el;
}

/** Changes an element's text with a quick fade-out / spring-in, never a single-frame swap. */
const pending = new WeakMap<HTMLElement, string>();
export function swapText(el: HTMLElement, next: string) {
  if ((pending.get(el) ?? el.textContent) === next) return;
  pending.set(el, next);
  if (!el.textContent) {
    el.textContent = next;
    return;
  }
  el.animate([{ opacity: 1, transform: "none" }, { opacity: 0, transform: "translateY(-3px)" }], {
    duration: 120,
    easing: "ease-in",
    fill: "forwards",
  }).onfinish = () => {
    if (pending.get(el) !== next) return;
    el.textContent = next;
    el.animate([{ opacity: 0, transform: "translateY(3px)" }, { opacity: 1, transform: "none" }], {
      duration: 420,
      easing: SPRING,
      fill: "forwards",
    });
  };
}

/** Removes an element after collapsing it, instead of yanking it out of the layout. */
export function collapseAway(el: HTMLElement) {
  const hgt = el.offsetHeight;
  el.animate([{ opacity: 1, height: hgt + "px" }, { opacity: 0, height: "0px", marginTop: "0px", marginBottom: "0px", paddingTop: "0px", paddingBottom: "0px" }], {
    duration: 320,
    easing: "cubic-bezier(.3,.7,.2,1)",
  }).onfinish = () => el.remove();
}

export function fmtNum(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + " K";
  return String(n);
}

export const fmtBytes = (b: number) => (b / 2 ** 30).toFixed(1) + " GB";

/** Background for a token: clear when the model was sure, warmer and redder as it was less sure. */
export function confidenceColor(logprob: number): string {
  const p = Math.exp(logprob);
  if (p > 0.9) return "transparent";
  if (p > 0.5) return `rgba(210,177,92,${(0.9 - p) * 0.9})`;
  return `rgba(214,122,106,${0.3 + (0.5 - p) * 0.6})`;
}

export function pct(logprob: number): string {
  const p = Math.exp(logprob) * 100;
  return p >= 99.95 ? "100%" : p < 0.1 ? "<0.1%" : p.toFixed(1) + "%";
}

export function showToken(text: string): string {
  return JSON.stringify(text).slice(1, -1);
}
