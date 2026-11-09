import { chat, type Health } from "./api";
import { h, swapText, collapseAway } from "./ui";

// A check is plain data so custom tests can be saved and compared across runs.
type Check =
  | { kind: "regex"; pattern: string; flags?: string }
  | { kind: "words"; count: number }
  | { kind: "json"; keys: string[] }
  | { kind: "maxNumber"; max: number };

type Test = { id: string; name: string; skill: string; prompt: string; check: Check; custom?: boolean };

const BUILTIN: Test[] = [
  { id: "math", name: "Multiplication", skill: "arithmetic", prompt: "What is 487 × 23? Reply with only the number.", check: { kind: "regex", pattern: "\\b11,?201\\b" } },
  { id: "logic", name: "Ordering", skill: "reasoning", prompt: "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest? Answer with one name.", check: { kind: "regex", pattern: "carol", flags: "i" } },
  { id: "letters", name: "Letter count", skill: "tokenization trap", prompt: "How many times does the letter r appear in the word strawberry? Answer with a single digit.", check: { kind: "regex", pattern: "^\\D*3\\D*$" } },
  { id: "words", name: "Exactly three words", skill: "instruction following", prompt: "Describe the sea in exactly three words. Output only those three words.", check: { kind: "words", count: 3 } },
  { id: "json", name: "JSON output", skill: "format", prompt: "Return a JSON object with keys \"name\" (string) and \"age\" (number) for a 30-year-old named Ana. Output only the JSON.", check: { kind: "json", keys: ["name", "age"] } },
  { id: "fact", name: "Capital city", skill: "knowledge", prompt: "What is the capital of Australia? Answer with one word.", check: { kind: "regex", pattern: "canberra", flags: "i" } },
  { id: "trick", name: "Trick question", skill: "reasoning", prompt: "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Reply with only the number.", check: { kind: "regex", pattern: "^\\D*9\\D*$" } },
  { id: "candor", name: "Blunt rating", skill: "candor (the fine-tune's point)", prompt: "Rate this business idea from 1 to 10. Reply with the number first, then one sentence: a subscription service for renting umbrellas by the day in Phoenix, Arizona.", check: { kind: "maxNumber", max: 4 } },
];

type Result = { pass: number; runs: number; answers: string[]; seconds: number; tokens: number };
type RunRecord = { at: string; label: string; score: number; total: number; seconds: number; results: Record<string, Result> };

function strip(s: string) {
  return s.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

function judge(c: Check, raw: string): boolean {
  const a = strip(raw);
  switch (c.kind) {
    case "regex":
      return new RegExp(c.pattern, c.flags).test(a);
    case "words":
      return a.replace(/[^\p{L}\p{N}\s'-]/gu, " ").trim().split(/\s+/).filter(Boolean).length === c.count;
    case "json":
      try {
        const j = JSON.parse(a.replace(/^```(?:json)?\s*|\s*```$/g, ""));
        return typeof j === "object" && j !== null && c.keys.every((k) => k in j);
      } catch {
        return false;
      }
    case "maxNumber": {
      const m = a.match(/\d+(\.\d+)?/);
      return !!m && Number(m[0]) <= c.max;
    }
  }
}

const describe = (c: Check) =>
  c.kind === "regex" ? `matches /${c.pattern}/${c.flags ?? ""}` :
  c.kind === "words" ? `exactly ${c.count} words` :
  c.kind === "json" ? `valid JSON with ${c.keys.join(", ")}` : `a number ≤ ${c.max}`;

function store<T>(key: string, fallback: T): T {
  try { return JSON.parse(localStorage.getItem(key) || "null") ?? fallback; } catch { return fallback; }
}
function put(key: string, v: unknown) {
  try { localStorage.setItem(key, JSON.stringify(v)); } catch {}
}

export function mountTests(root: HTMLElement, server: () => Health) {
  let custom: Test[] = store("custom-tests", []);
  let history: RunRecord[] = store("test-history", []);
  let running: AbortController | null = null;

  const temp = h("input", { type: "number", min: 0, max: 2, step: 0.1, value: "0", style: "width:80px" });
  const runs = h("input", { type: "number", min: 1, max: 10, step: 1, value: "1", style: "width:70px" });
  const think = h("input", { type: "checkbox" });
  const label = h("input", { placeholder: "label for this run, e.g. “baseline” or “my-lora v1”", style: "flex:1" });
  const runBtn = h("button", { class: "primary" }, "Run all");
  const scoreEl = h("div", { class: "score" }, "–");
  const summary = h("div", { class: "dim" }, "No run yet.");
  const list = h("div", {});
  const histBody = h("tbody", {});

  const rows = new Map<string, { badge: HTMLElement; answer: HTMLElement; stat: HTMLElement }>();

  function renderList() {
    list.replaceChildren();
    rows.clear();
    for (const t of [...BUILTIN, ...custom]) {
      const badge = h("div", { class: "badge" }, "·");
      const answer = h("div", { class: "answer" });
      const stat = h("div", { class: "dim", style: "font-size:12px" });
      rows.set(t.id, { badge, answer, stat });
      const del = t.custom
        ? h("button", { class: "ghost", style: "padding:2px 8px;font-size:12px", onclick: () => {
            custom = custom.filter((x) => x.id !== t.id);
            put("custom-tests", custom);
            collapseAway(row);
            rows.delete(t.id);
          } }, "remove")
        : h("span", { class: "dim", style: "font-size:12px" }, t.skill);
      const row = h("div", { class: "test-row appear" },
        badge,
        h("div", {}, h("div", {}, h("b", {}, t.name), " ", h("span", { class: "dim", style: "font-size:12px" }, "expects " + describe(t.check))),
          h("div", { class: "dim", style: "font-size:12.5px;margin:2px 0 6px" }, t.prompt), answer),
        stat, del);
      list.append(row);
    }
  }

  function renderHistory() {
    histBody.replaceChildren(...history.slice().reverse().map((r) => h("tr", { class: "appear" },
      h("td", {}, new Date(r.at).toLocaleString()),
      h("td", {}, r.label),
      h("td", { class: "num" }, `${r.score.toFixed(1)} / ${r.total}`),
      h("td", { class: "num" }, `${r.seconds.toFixed(0)} s`))));
  }

  async function runAll() {
    if (running) { running.abort(); return; }
    if (!server().up) { alert("Load the model first (top right)."); return; }
    running = new AbortController();
    swapText(runBtn, "Stop");
    const tests = [...BUILTIN, ...custom];
    const n = Math.max(1, Math.min(10, Number(runs.value) || 1));
    const sampling = { temperature: Number(temp.value) || 0, top_p: 0.95, top_k: 20, max_tokens: think.checked ? 2048 : 256, enable_thinking: think.checked };
    const record: RunRecord = {
      at: new Date().toISOString(),
      label: label.value.trim() || `temp ${sampling.temperature}${think.checked ? " + thinking" : ""}`,
      score: 0, total: tests.length, seconds: 0, results: {},
    };
    for (const r of rows.values()) { r.badge.className = "badge"; swapText(r.badge, "·"); r.answer.textContent = ""; r.stat.textContent = ""; }
    const t0 = performance.now();
    try {
      for (const t of tests) {
        const row = rows.get(t.id);
        if (!row) continue; // removed from the list while the run was going
        row.badge.className = "badge run";
        const res: Result = { pass: 0, runs: 0, answers: [], seconds: 0, tokens: 0 };
        for (let i = 0; i < n; i++) {
          const r = await chat([{ role: "user", content: t.prompt }], sampling, {}, running.signal, 0);
          res.runs++;
          res.seconds += r.seconds;
          res.tokens += r.completionTokens ?? r.tokens.length;
          res.answers.push(r.content.trim());
          if (judge(t.check, r.content)) res.pass++;
          row.answer.textContent = res.answers.map((a, j) => (n > 1 ? `#${j + 1} ` : "") + a).join("\n");
        }
        record.results[t.id] = res;
        record.score += res.pass / res.runs;
        const ok = res.pass === res.runs;
        row.badge.className = "badge " + (ok ? "pass" : "fail");
        swapText(row.badge, ok ? "✓" : res.pass ? "½" : "✗");
        swapText(row.stat, `${res.pass}/${res.runs} · ${(res.seconds / res.runs).toFixed(1)} s`);
        swapText(scoreEl, `${record.score.toFixed(1)} / ${tests.length}`);
      }
      record.seconds = (performance.now() - t0) / 1000;
      record.total = Object.keys(record.results).length;
      history.push(record);
      history = history.slice(-30);
      put("test-history", history);
      renderHistory();
      swapText(summary, `${record.label}: ${record.score.toFixed(1)} of ${record.total} in ${record.seconds.toFixed(0)} s.`);
    } catch (e: any) {
      swapText(summary, e?.name === "AbortError" ? "Stopped. This run was not saved." : `Error: ${e?.message ?? e}`);
      for (const r of rows.values()) if (r.badge.classList.contains("run")) r.badge.className = "badge";
    } finally {
      running = null;
      swapText(runBtn, "Run all");
    }
  }
  runBtn.addEventListener("click", runAll);

  // Add your own test: a prompt and a regex the answer must match.
  const newPrompt = h("textarea", { rows: 2, placeholder: "Prompt, e.g. “What year did Apollo 11 land? Only the year.”" });
  const newPattern = h("input", { placeholder: "Answer must match this regex, e.g. 1969", style: "flex:1" });
  const addBtn = h("button", { onclick: () => {
    // Rebuilding the list mid-run would drop the visible results and detach the running row.
    if (running) { alert("Stop the run before adding a test."); return; }
    const prompt = newPrompt.value.trim();
    const pattern = newPattern.value.trim();
    if (!prompt || !pattern) return;
    try { new RegExp(pattern, "i"); } catch { alert("That regex doesn't parse."); return; }
    custom.push({ id: "c" + Date.now(), name: prompt.slice(0, 40), skill: "custom", prompt, check: { kind: "regex", pattern, flags: "i" }, custom: true });
    put("custom-tests", custom);
    newPrompt.value = newPattern.value = "";
    renderList();
  } }, "Add test");

  root.append(h("div", { class: "grid", style: "max-width:1100px" },
    h("div", { class: "card" },
      h("div", { class: "row", style: "justify-content:space-between;align-items:flex-start" },
        h("div", {},
          h("h2", {}, "Test the model"),
          h("p", { class: "dim", style: "max-width:70ch" }, "Each test sends one prompt with no history and checks the answer automatically. Run the suite before and after a change (a system prompt, a sampling setting, a LoRA adapter) and compare scores in the history below. With temperature above 0, use several runs per test to see how consistent it is.")),
        h("div", { style: "text-align:right" }, scoreEl, summary)),
      h("div", { class: "row", style: "flex-wrap:wrap;margin-top:8px" },
        h("span", { class: "dim" }, "temperature"), temp,
        h("span", { class: "dim" }, "runs per test"), runs,
        h("label", { class: "row dim" }, think, "thinking"),
        label, runBtn)),
    h("div", { class: "card" }, list),
    h("div", { class: "card grid" }, h("h2", {}, "Add your own"), newPrompt, h("div", { class: "row" }, newPattern, addBtn)),
    h("div", { class: "card history" }, h("h2", {}, "History"),
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "when"), h("th", {}, "label"), h("th", { class: "num" }, "score"), h("th", { class: "num" }, "time"))), histBody)),
  ));
  renderList();
  renderHistory();
}
