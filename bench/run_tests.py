#!/usr/bin/env python3
"""Re-runnable live tests for lowctx against the local model server (one request at a time).

    bench/run_tests.py [t1] [biz] [research] [long] [paste]     (default: all)

Logs go to bench/logs/<name>.log. Each test prints PASS/FAIL checks plus steps / seconds / max prompt tokens.
"""
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv/bin/python"
PY = str(PY if PY.exists() else sys.executable)
LOGS = ROOT / "bench/logs"
LOGS.mkdir(exist_ok=True)
BUDGET = 12000
TIMEOUT = 2400  # seconds per test


def run(name, cwd, stdin, extra=()):
    t = time.time()
    try:  # subprocess.run kills the child on timeout, so no orphans
        p = subprocess.run([PY, str(ROOT / "lowctx.py"), *extra], cwd=cwd, input=stdin, capture_output=True,
                           text=True, timeout=TIMEOUT)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += "\n[TIMEOUT: killed]"
    (LOGS / f"{name}.log").write_text(out)
    pts = [int(x) for x in re.findall(r"prompt=(\d+) tok", out)]
    stats = dict(secs=time.time() - t, steps=len(pts), max_prompt=max(pts or [0]),
                 compactions=len(re.findall(r"\[compress via", out)), out=out)
    return stats


def report(name, st, checks):
    ok = all(c for _, c in checks)
    print(f"\n== {name}: {'PASS' if ok else 'FAIL'}  steps={st['steps']} secs={st['secs']:.0f} "
          f"max_prompt={st['max_prompt']} compressor_calls={st['compactions']}")
    for label, c in checks:
        print(f"   [{'ok' if c else 'XX'}] {label}")
    return ok


def t1():
    # work on a throwaway copy so the planted bugs in bench/t1 stay intact
    cwd = Path(tempfile.mkdtemp(prefix="porthole-t1-")) / "t1"
    shutil.copytree(ROOT / "bench/t1", cwd)
    task = ("Tests in tests/ are failing. Fix the bugs in the inventory/ code (not the tests). "
            f"Run tests with: {PY} -m pytest -q")
    st = run("t1", cwd, task + "\n/quit\n", ["--mode", "code"])
    r = subprocess.run([PY, "-m", "pytest", "-q"], cwd=cwd, capture_output=True, text=True)
    shutil.rmtree(cwd.parent, ignore_errors=True)
    return report("t1 coding regression", st, [("pytest passes after run", r.returncode == 0),
                                              (f"max prompt <= {BUDGET + 1500}", st["max_prompt"] <= BUDGET + 1500)])


def biz():
    q = "Read my business files and be brutally honest: what's wrong with this business and what should I do in the next 30 days?"
    st = run("biz", ROOT / "bench/biz", q + "\n/quit\n")
    o = st["out"]
    return report("advisor on business files", st, [
        ("read business_plan.md", "business_plan.md" in o), ("read revenue.csv", "revenue.csv" in o),
        ("mentions churn", "churn" in o.lower()), ("mentions Slack/integration", "slack" in o.lower()),
        ("mentions price/pricing", re.search(r"\$9|underpric|pricing", o, re.I) is not None)])


def research():
    q = ("Use web research (search, then fetch real pages, don't rely on snippets): what do Teamwork.com and Productive.io "
         "charge per user per month today? Give exact plan prices and cite the URLs you fetched. Then say in 2 sentences "
         "whether my $9/user plan is underpriced.")
    st = run("research", ROOT / "bench/biz", q + "\n/quit\n")
    o = st["out"]
    return report("web research", st, [("called web_search", "→ web_search" in o), ("called fetch", "→ fetch" in o),
                                       ("cites a URL", re.search(r"https?://\S+", o.split("→ fetch")[-1]) is not None)])


LONG = [
    "Read business_plan.md and revenue.csv. What is the single biggest problem? Two sentences.",
    "What was the total number of customers lost over the 12 months in revenue.csv? Use bash/python to compute if needed.",
    "Now read pitch_notes.txt. How does what the founder tells investors about churn contradict the data?",
    "Search the web for typical monthly churn for SMB SaaS and tell me in one sentence how we compare.",
    "What was the churn number you found earlier in revenue.csv (customers lost in the last month listed)?",
    "What is the ask in the pitch notes, exactly?",
    "Is $9 per user a good price? Compare it with what Trello or Asana charge if you know; say if you're guessing.",
    "What did the cost-to-serve per user come to in the plan, and what is the resulting gross margin at $9?",
    "Who is the one part-time contractor and what does he cost per month?",
    "Remind me: what were the three integrations missing according to the exit survey?",
    "Given everything, rank my top 3 actions for this week, one line each.",
    "Which month in revenue.csv had the highest support_hours, and how many?",
    "What was my personal runway again, and what's the monthly burn implied by the contractor alone?",
    "Last question: summarize in 3 bullets what you told me about churn over this whole conversation.",
]


def long():
    st = run("long", ROOT / "bench/biz", "\n".join(LONG) + "\n/notes\n/quit\n", ["--budget", str(BUDGET)])
    o = st["out"]
    turns = re.findall(r"\[turn: .*?\]", o)
    ans = o.split("you> ")
    def reply(i):  # text between prompt i and the next
        return ans[i + 1] if i + 1 < len(ans) else ""
    return report("long 14-turn session", st, [
        (f"{len(turns)} turns completed", len(turns) == len(LONG)),
        ("compaction happened", st["compactions"] >= 1),
        ("T2 total lost = 76", "76" in reply(1)),
        ("T5 recalls 6 lost in last month (2026-10)", re.search(r"\b6\b", reply(4)) is not None),
        ("T6 ask $150k", "150" in reply(5)),
        ("T10 Slack, Google Drive, QuickBooks", all(k in reply(9) for k in ("Slack", "Drive", "QuickBooks"))),
        ("T12 support peak 2026-08 / 52", "52" in reply(11)),
        (f"max prompt <= {BUDGET + 1500}", st["max_prompt"] <= BUDGET + 1500)])


def paste():
    words = ("Our company sells inventory software to hardware stores. Last quarter we lost two key clients because "
             "onboarding took six weeks. ").split()
    body = []
    for i in range(1, 121):
        body.append(f"Paragraph {i}: " + " ".join(words) + f" Detail #{i}: the cohort {i} paid ${100 + i} per month.")
    body.insert(77, "CRITICAL FACT: the board decided to freeze hiring until the churn drops below 4 percent.")
    text = "\n".join(body)
    msg = '"""Here is my internal memo, read it carefully.\n' + text + '\n\nQuestion: in one sentence, what is the single actionable decision in this memo?"""'
    st = run("paste", ROOT / "bench/biz", msg + "\n/quit\n")
    saved = sorted((ROOT / "bench/biz/.lowctx").glob("input-*.txt"))
    return report(f"big paste ({len(msg)} chars)", st, [
        ("full text saved to .lowctx/input-*.txt", bool(saved) and "CRITICAL FACT" in saved[-1].read_text()),
        ("briefing happened", "large message" in st["out"]),
        ("answer mentions hiring freeze", re.search(r"hiring|freeze", st["out"].split("[large message")[-1], re.I) is not None),
        (f"max prompt <= {BUDGET + 1500}", st["max_prompt"] <= BUDGET + 1500)])


if __name__ == "__main__":
    names = sys.argv[1:] or ["t1", "biz", "research", "long", "paste"]
    res = {n: globals()[n]() for n in names}
    print("\nSUMMARY:", ", ".join(f"{n}={'PASS' if v else 'FAIL'}" for n, v in res.items()))
    sys.exit(0 if all(res.values()) else 1)
