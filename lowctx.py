#!/usr/bin/env python3
"""lowctx: chat + agent harness for small-context / slow local models.

The local model (any OpenAI-compatible endpoint) only ever sees a small, rebuilt prompt:

    system (+ conversation notes)  +  recent turns  +  current turn's tool exchanges

Everything else lives outside the prompt:
  * older turns / exchanges are folded into "notes" by a compressor model
    (DeepSeek when DEEPSEEK_API_KEY is set, otherwise the local model itself)
  * big tool outputs / web pages are saved to .lowctx/ and replaced by a question-focused digest
  * huge user pastes are saved to disk and replaced by a brief
  * file reads are paged (line numbers + ranges) instead of dumped whole

Modes: advisor (default; blunt business/strategy/code advisor) and code (coding agent).
Stdlib only.   Usage:  lowctx.py [--mode advisor|code] [--cwd DIR] [--budget 5000] ["first message"]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from html.parser import HTMLParser
from pathlib import Path

# ---------------------------------------------------------------- config

# .env next to this file (KEY=VALUE lines); real environment variables win
_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _k, _, _v = _line.strip().partition("=")
        if _k and not _k.startswith("#") and _v:
            os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))

LOCAL_URL = os.environ.get("LOWCTX_URL", "http://127.0.0.1:8080/v1")
# must match the build start-server.sh loads (QWEN_QUANT, default 4-bit)
DEFAULT_MODEL = str(Path(__file__).resolve().parent / "Qwen3.8-27B-Uncensored-MLX" / os.environ.get("QWEN_QUANT", "4-bit"))
LOCAL_MODEL = os.environ.get("LOWCTX_MODEL", "")  # empty -> autodetect from /models
LOCAL_KEY = os.environ.get("LOWCTX_KEY", "")
THINK_FIELD = os.environ.get("LOWCTX_THINK_FIELD", "1") != "0"  # send enable_thinking at all?

COMP_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
COMP_URL = os.environ.get("DEEPSEEK_URL", "https://api.deepseek.com/v1")
COMP_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

BASH_TIMEOUT = int(os.environ.get("LOWCTX_BASH_TIMEOUT", "120"))
READ_LINES = 150  # max lines per read call
READ_CHARS = 3500  # max chars per read call
INLINE_CHARS = 2400  # tool output above this gets digested
MSG_CAP = 6000  # hard cap for any single stored message (chars)
OLD_TOOL_CAP = 700  # tool results of finished turns are clipped to this
KEEP_RECENT = 2  # tool exchanges of the current turn always kept verbatim
MAX_CALLS_PER_STEP = 4
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

SYSTEM_CODE = """You are a coding agent working in {cwd}. You act only through tools.
Rules:
- Look before you change: ls/grep/read the relevant code first.
- read returns numbered lines; use start= to page. Never guess file contents.
- Prefer edit (exact unique snippet replace) over rewriting whole files.
- After changing code, run it or its tests with bash to verify.
- One small step per tool call. Do not repeat a call that already succeeded.
- Earlier work is summarized in "Notes"; trust them.
- delegate(task) sends docs/API lookups to a cloud subagent; if it declines, do it yourself.
- When the request is done (or impossible), reply with a short plain-text summary and NO tool call."""

SYSTEM_ADVISOR = """You are a blunt, candid advisor to the user on their business, code and strategy. Working dir: {cwd}.
- No flattery, hedging, disclaimers or motivational fluff. Weakest points and biggest risks first, then what to do.
- Be concrete: numbers, named next actions with deadlines. If something is a bad idea, say so plainly and why. Disagree when warranted. Say "I don't know" when you don't.
- Ground claims in files you read or pages you fetched; cite file names/URLs and quote figures exactly. Label guesses as guesses. Never invent data: read or search first.
- Use tools: ls/read the user's files (page with start=), web_search then fetch the best pages (snippets are not enough).
- Web research: call delegate(task) FIRST, one call with a complete task (what to find, which sources, what figures). Then judge its report yourself. Only use web_search/fetch yourself if delegate declines/fails or for one quick check.
- Earlier turns are summarized in "Notes"; trust them for follow-ups.
- Make a few tool calls, then answer in plain text with NO tool call. Be compact unless asked for depth."""

TOOLS = {
    "ls": ("List a directory.", {"path": "string"}, []),
    "read": ("Read a text/markdown/csv/pdf/docx file with line numbers; page with start= (1-based).",
             {"path": "string", "start": "integer"}, ["path"]),
    "grep": ("Regex search in files; returns path:line:text.", {"pattern": "string", "path": "string"}, ["pattern"]),
    "edit": ("Replace one exact, unique snippet `old` in a file with `new`.",
             {"path": "string", "old": "string", "new": "string"}, ["path", "old", "new"]),
    "write": ("Create or overwrite a whole file.", {"path": "string", "content": "string"}, ["path", "content"]),
    "bash": ("Run a shell command in the working dir.", {"cmd": "string"}, ["cmd"]),
    "web_search": ("Web search; returns top results (title, url, snippet). For multi-source research use delegate instead.",
                   {"query": "string"}, ["query"]),
    "fetch": ("Download a URL and return its text (big pages are digested; full text saved to a file).",
              {"url": "string"}, ["url"]),
    "delegate": ("PREFERRED for web research: a fast cloud subagent searches, fetches and reads many pages/files "
                 "and returns a short sourced report. Write the task fully: it can't see this chat.",
                 {"task": "string"}, ["task"]),
}
MODE_TOOLS = {
    "code": ["ls", "read", "grep", "edit", "write", "bash", "delegate"],
    "advisor": ["ls", "read", "grep", "bash", "web_search", "fetch", "delegate"],
}


def schemas(mode: str) -> list:
    names = MODE_TOOLS[mode]
    if not COMP_KEY:
        return schemas_for([n for n in names if n != "delegate"])
    # list delegate first so the small model reaches for it before doing research by hand
    return schemas_for(["delegate"] + [n for n in names if n != "delegate"])


def schemas_for(names: list) -> list:
    out = []
    for n in names:
        d, props, req = TOOLS[n]
        out.append({"type": "function", "function": {
            "name": n, "description": d,
            "parameters": {"type": "object", "properties": {k: {"type": t} for k, t in props.items()},
                           "required": req}}})
    return out


NOTES_CODE = """You maintain the working memory of a coding agent whose context window is tiny.
Merge OLD NOTES with the NEW EVENTS into updated notes. The agent will see ONLY these notes
plus its last couple of steps, so keep everything it needs to continue correctly:
- goal and any user constraints
- facts discovered: exact file paths, function/class names, line numbers, key code snippets,
  error messages (verbatim if short), commands that work
- changes made so far (file + what changed) and whether they were verified
- dead ends already tried (so they are not repeated)
- current status and the most sensible next step
Be dense; bullet points; no chatter. Hard limit ~{words} words. Output only the notes."""

NOTES_CHAT = """You maintain the memory of an advisor AI in a long conversation with a user. The advisor sees ONLY
these notes plus the latest turns, so keep what it needs to answer follow-ups correctly:
- who the user is, their business/project, goals, constraints, preferences
- every concrete fact and number found (figures, prices, names, dates) WITH its source (file path or URL), verbatim
- files read / pages fetched (paths, URLs) and what was in them
- advice and conclusions already given, and the user's decisions/objections
- open questions and unfinished tasks
Merge OLD NOTES with NEW EVENTS; drop what is superseded. Dense bullets, no chatter.
Hard limit ~{words} words. Output only the notes."""

DIGEST_PROMPT = """An AI assistant with a tiny context window ran a tool. The raw output is too big for it.
User's current question: {goal}
Tool call: {call}
Write a digest of the output containing ONLY what matters for that question: exact figures, prices,
names, dates, key claims and short verbatim quotes, errors/tracebacks (key lines), relevant lines with
line numbers, summary counts. Say if the output does not contain what the question needs.
Under {words} words. Output only the digest."""

BRIEF_PROMPT = """Condense this user message for an AI assistant with a tiny context window.
Keep every concrete requirement, question, name, number, constraint, and short snippet verbatim.
Drop repetition and filler. Under {words} words. Output only the condensed message.

MESSAGE:
{text}"""


# ---------------------------------------------------------------- llm clients


class HTTPFail(RuntimeError):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def _req(url: str, key: str, body: dict) -> urllib.request.Request:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers)


def post_chat(url: str, key: str, body: dict, timeout: int = 1800) -> dict:
    """Non-streaming chat call. Retries network errors only; HTTP errors raise HTTPFail."""
    req = _req(url, key, body)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise HTTPFail(e.code, f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:300]}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < 2:
                time.sleep(3)
                continue
            raise HTTPFail(0, f"cannot reach {url}: {e}") from e
    raise HTTPFail(0, "unreachable")


def stream_chat(url: str, key: str, body: dict, on_text, timeout: int = 1800) -> dict:
    """Streaming chat call; returns a dict shaped like a non-streaming response."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = _req(url, key, body)
    content, calls, usage, finish = [], {}, {}, None
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise HTTPFail(e.code, f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:300]}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise HTTPFail(0, f"cannot reach {url}: {e}") from e
    with r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ch = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ch.get("usage"):
                usage = ch["usage"]
            for c in ch.get("choices") or []:
                finish = c.get("finish_reason") or finish
                d = c.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                    on_text(d["content"])
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    a = fn.get("arguments")
                    slot["arguments"] += a if isinstance(a, str) else (json.dumps(a) if a else "")
    tcs = [{"id": s["id"] or uuid.uuid4().hex[:12], "type": "function",
            "function": {"name": s["name"], "arguments": s["arguments"] or "{}"}}
           for _, s in sorted(calls.items())]
    msg = {"role": "assistant", "content": "".join(content) or None}
    if tcs:
        msg["tool_calls"] = tcs
    return {"choices": [{"message": msg, "finish_reason": finish}], "usage": usage}


class Streamer:
    """Prints streamed text, but hides <think> blocks and tool-call XML."""

    def __init__(self, out=sys.stdout):
        self.buf, self.mode, self.out, self.printed = "", "probe", out, False

    def _emit(self, s: str) -> None:
        if not self.printed:
            s = s.lstrip()
        if s:
            self.out.write(s)
            self.out.flush()
            self.printed = True

    def feed(self, t: str) -> None:
        if self.mode == "stream":
            return self._emit(t)
        self.buf += t
        while True:
            if self.mode == "think":
                i = self.buf.find("</think>")
                if i < 0:
                    return
                self.buf, self.mode = self.buf[i + 8:], "probe"
            if self.mode == "hide":
                return
            s = self.buf.lstrip()
            if not s:
                return
            for tag in ("<think>", "<tool_call>", "<function="):
                if s.startswith(tag):
                    self.mode = "think" if tag == "<think>" else "hide"
                    break
                if tag.startswith(s):
                    return  # still ambiguous
            else:
                self.mode = "stream"
                b, self.buf = self.buf, ""
                return self._emit(b)
            if self.mode == "hide":
                return


class Compressor:
    """Cheap big-context model used only to summarize. Falls back to the local model."""

    def __init__(self, log, local_model):
        self.log, self.local_model = log, local_model
        self.use_remote = bool(COMP_KEY)
        self.name = COMP_MODEL if self.use_remote else "local"

    @property
    def max_in(self) -> int:  # chars the compressor can take
        return 200_000 if self.use_remote else 14_000

    def _call(self, remote: bool, prompt: str, words: int) -> str:
        url, key, model = (COMP_URL, COMP_KEY, COMP_MODEL) if remote else (LOCAL_URL, LOCAL_KEY, self.local_model())
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(words * 2.2) + 64, "temperature": 0.1}
        if remote:
            # deepseek-flash is a reasoning model; without this it spends max_tokens thinking and returns ""
            body["thinking"] = {"type": "disabled"}
        elif THINK_FIELD:
            body["enable_thinking"] = False
        d = post_chat(url, key, body, timeout=180 if remote else 1800)
        out = (d["choices"][0]["message"].get("content") or "").strip()
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()

    def run(self, prompt: str, words: int, goal: str = "") -> str:
        t = time.time()
        if len(prompt) > self.max_in:
            prompt = focus(prompt, goal, self.max_in)
        out, via = "", self.name
        if self.use_remote:
            try:
                out = self._call(True, prompt, words)
            except Exception as e:  # noqa: BLE001
                self.log(f"  [compressor {COMP_MODEL} failed ({str(e)[:80]}); using local model]")
                via = "local(fallback)"
                if len(prompt) > 14_000:
                    prompt = focus(prompt, goal, 14_000)
        if not out:
            out = self._call(False, prompt, words)
        self.log(f"  [compress via {via}: {len(prompt)}→{len(out)} chars, {time.time()-t:.1f}s]")
        return out


def focus(text: str, goal: str, limit: int) -> str:
    """Pick the most goal-relevant lines of a long text (cheap, no model)."""
    if len(text) <= limit:
        return text
    lines = [l[:400] for l in text.splitlines() if l.strip()]
    kw = set(re.findall(r"[a-z0-9$%]{4,}", goal.lower()))
    base = []
    for i, l in enumerate(lines):
        ll = l.lower()
        base.append(2 * sum(k in ll for k in kw) + 0.3 * min(len(re.findall(r"\d", l)), 6) + (3 if i < 6 else 0))
    price = [bool(re.search(r"[$€£]\s?\d|\d\s?(usd|eur)|/ ?mo\b|per (user|month|seat)", l.lower())) for l in lines]
    scored = []
    for i in range(len(lines)):
        s2 = base[i] + (3 if price[i] else 0) + (2 if any(price[j] for j in range(max(0, i - 2), min(len(lines), i + 2))) else 0)
        scored.append((s2, i))
    chosen, used = set(), 0
    for s, i in sorted(scored, key=lambda x: -x[0]):
        if used + len(lines[i]) + 1 > limit - 100:
            continue
        chosen.add(i)
        used += len(lines[i]) + 1
    out, prev = [], -1
    for i in sorted(chosen):
        if prev >= 0 and i != prev + 1:
            out.append("…")
        out.append(lines[i])
        prev = i
    return "\n".join(out)


# ---------------------------------------------------------------- web helpers


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "head", "iframe"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
             "table", "ul", "ol", "header", "footer", "blockquote", "pre", "dt", "dd", "hr", "main"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip, self.title, self.in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        if tag in self.SKIP:
            self.skip += 1
        if tag in ("td", "th"):
            self.parts.append(" | ")
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, d):
        if self.in_title:
            self.title += d
        elif not self.skip:
            self.parts.append(d)


def html_to_text(src: str) -> str:
    p = _Text()
    try:
        p.feed(src)
        p.close()
    except Exception:  # noqa: BLE001
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = "\n".join(l.strip() for l in text.split("\n") if l.strip())
    title = " ".join(p.title.split())
    return (f"# {title}\n\n" if title else "") + text


def http_get(url: str, data: bytes | None = None, timeout: int = 25, max_bytes: int = 3_000_000):
    req = urllib.request.Request(url, data, {
        "User-Agent": UA, "Accept": "text/html,application/xhtml+xml,text/plain,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(max_bytes)
        if r.headers.get("Content-Encoding") == "gzip":
            try:
                raw = gzip.decompress(raw)
            except Exception:  # noqa: BLE001
                pass
        return raw, r.headers.get("Content-Type", ""), r.geturl()


def decode_body(raw: bytes, ctype: str) -> str:
    m = re.search(r"charset=([\w-]+)", ctype or "", re.I) or re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", raw[:4096], re.I)
    enc = (m.group(1).decode() if m and isinstance(m.group(1), bytes) else (m.group(1) if m else "utf-8"))
    try:
        return raw.decode(enc, "replace")
    except LookupError:
        return raw.decode("utf-8", "replace")


def _clean(s: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", s)).split())


def search_ddg(q: str) -> list[tuple[str, str, str]]:
    raw, ct, _ = http_get("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(q))
    page = decode_body(raw, ct)
    out = []
    for blk in re.split(r'<div class="result[ "]', page)[1:]:
        m = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        href = html.unescape(m.group(1))
        if "uddg=" in href:
            href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0])
        if "duckduckgo.com/y.js" in href or href.startswith("//duckduckgo"):
            continue  # ad
        s = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', blk, re.S)
        out.append((_clean(m.group(2)), href, _clean(s.group(1)) if s else ""))
    return out


def search_bing(q: str) -> list[tuple[str, str, str]]:
    raw, ct, _ = http_get("https://www.bing.com/search?q=" + urllib.parse.quote_plus(q))
    page = decode_body(raw, ct)
    out = []
    for blk in re.split(r'<li class="b_algo"', page)[1:]:
        m = re.search(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        href = html.unescape(m.group(1))
        if "bing.com/ck/a" in href:
            u = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("u", [""])[0]
            if u.startswith("a1"):
                try:
                    import base64
                    href = base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode()
                except Exception:  # noqa: BLE001
                    pass
        s = re.search(r"<p[^>]*>(.*?)</p>", blk, re.S)
        out.append((_clean(m.group(2)), href, _clean(s.group(1)) if s else ""))
    return out


# ---------------------------------------------------------------- tools


def _has(binary: str) -> bool:
    return shutil.which(binary) is not None


class Tools:
    def __init__(self, cwd: Path, store: Path):
        self.cwd, self.store = cwd, store

    def path(self, p: str) -> Path:
        q = Path(p or ".").expanduser()
        return (q if q.is_absolute() else self.cwd / q).resolve()

    def run(self, name: str, a: dict) -> str:
        try:
            fn = getattr(self, "t_" + name, None)
            if fn is None:
                return f"error: unknown tool {name!r}. Tools: " + ", ".join(TOOLS)
            if not isinstance(a, dict):
                return "error: arguments must be a JSON object"
            return fn(**a)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"

    def t_ls(self, path: str = ".") -> str:
        p = self.path(path)
        if not p.exists():
            return f"error: {path} does not exist"
        if p.is_file():
            return f"{p.name}  ({p.stat().st_size}b) is a file"
        skip = {".git", "__pycache__", ".venv", "node_modules", ".lowctx"}
        items = sorted((x.name + "/" if x.is_dir() else f"{x.name}  ({x.stat().st_size}b)")
                       for x in p.iterdir() if x.name not in skip)
        return "\n".join(items[:200]) or "(empty)"

    def _doc_text(self, p: Path) -> str:
        """Text of a file; converts pdf/docx/etc. when a converter exists."""
        ext = p.suffix.lower()
        if ext == ".pdf":
            if not _has("pdftotext"):
                raise RuntimeError("cannot read PDF: `pdftotext` is not installed (brew install poppler). "
                                   "Ask the user to paste the text or convert it.")
            cache = self.store / f"conv-{hashlib.md5(str(p).encode()).hexdigest()[:8]}.txt"
            if not cache.exists() or cache.stat().st_mtime < p.stat().st_mtime:
                r = subprocess.run(["pdftotext", "-layout", str(p), str(cache)], capture_output=True, text=True, timeout=120)
                if r.returncode:
                    raise RuntimeError(f"pdftotext failed: {r.stderr[:200]}")
            return cache.read_text(errors="replace")
        if ext in (".docx", ".doc", ".rtf", ".odt", ".html", ".htm") and _has("textutil"):
            r = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(p)], capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                return r.stdout
        data = p.read_bytes()
        if b"\x00" in data[:4096]:
            raise RuntimeError(f"{p.name} looks like a binary file ({ext or 'no extension'}); cannot read it")
        text = data.decode("utf-8", "replace")
        if ext in (".html", ".htm"):
            text = html_to_text(text)
        return text

    def t_read(self, path: str, start: int = 1) -> str:
        p = self.path(path)
        if not p.exists():
            return f"error: {path} does not exist"
        if p.is_dir():
            return f"error: {path} is a directory; use ls"
        lines = self._doc_text(p).splitlines()
        start = max(1, int(start or 1))
        chunk, used = [], 0
        for i, l in enumerate(lines[start - 1:start - 1 + READ_LINES], start):
            l = l[:1500] + ("…" if len(l) > 1500 else "")
            if chunk and used + len(l) > READ_CHARS:
                break
            chunk.append(f"{i:>4}| {l}")
            used += len(l) + 7
        end = start + len(chunk) - 1
        tail = f"\n[lines {start}-{end} of {len(lines)}"
        tail += f"; read with start={end + 1} for more]" if end < len(lines) else "; end of file]"
        return ("\n".join(chunk) or "(empty file)") + tail

    def t_grep(self, pattern: str, path: str = ".") -> str:
        p = self.path(path)
        cmd = (["rg", "-n", "--no-heading", "-g", "!.lowctx", pattern, str(p)] if _has("rg") else
               ["grep", "-rnIE", "--exclude-dir=.git", "--exclude-dir=.venv", "--exclude-dir=.lowctx", pattern, str(p)])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = r.stdout.replace(str(self.cwd) + "/", "").splitlines()
        if not lines:
            return "no matches"
        more = f"\n[{len(lines) - 60} more matches; narrow the pattern/path]" if len(lines) > 60 else ""
        return "\n".join(l[:240] for l in lines[:60]) + more

    def t_edit(self, path: str, old: str, new: str) -> str:
        p = self.path(path)
        text = p.read_text(encoding="utf-8")
        n = text.count(old)
        if n == 1:
            p.write_text(text.replace(old, new, 1), encoding="utf-8")
            return f"ok: edited {path} at line {text[: text.index(old)].count(chr(10)) + 1}"
        if n > 1:
            return f"error: `old` occurs {n} times in {path}; include more surrounding lines to make it unique."
        first = next((l.strip() for l in old.splitlines() if l.strip()), "")
        lines = text.splitlines()
        hit = next((i for i, l in enumerate(lines) if first and first in l), None)
        hint = ""
        if hit is not None:
            lo = max(0, hit - 3)
            hint = "\nClosest match — actual text:\n" + "\n".join(f"{i + 1:>4}| {l}" for i, l in enumerate(lines[lo:hit + 8], lo))
        return f"error: `old` not found in {path} (whitespace must match exactly).{hint}"

    def t_write(self, path: str, content: str) -> str:
        p = self.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"ok: wrote {path} ({content.count(chr(10)) + 1} lines)"

    def t_bash(self, cmd: str) -> str:
        try:
            r = subprocess.run(cmd, shell=True, cwd=self.cwd, capture_output=True, text=True,
                               timeout=BASH_TIMEOUT, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return f"error: timed out after {BASH_TIMEOUT}s"
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
        return f"exit={r.returncode}\n{out.strip()}"

    def t_web_search(self, query: str) -> str:
        if not str(query).strip():
            return "error: empty query"
        res, err = [], ""
        for fn in (search_ddg, search_bing):
            try:
                res = fn(query)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                continue
            if res:
                break
        if not res:
            return f"no results ({err or 'search engines returned nothing'}). Try different keywords."
        return "\n".join(f"{i}. {t}\n   {u}\n   {s[:220]}" for i, (t, u, s) in enumerate(res[:6], 1))

    def t_delegate(self, task: str) -> str:
        if getattr(self, "in_delegate", False):
            return "error: delegate is not available inside a subagent"
        self.in_delegate = True
        try:
            return delegate_task(task, self, getattr(self, "log", print))
        finally:
            self.in_delegate = False

    def t_fetch(self, url: str) -> str:
        url = str(url).strip()
        if not re.match(r"https?://", url):
            url = "https://" + url
        try:
            raw, ct, final = http_get(url)
        except urllib.error.HTTPError as e:
            return f"error: HTTP {e.code} fetching {url}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return f"error: could not fetch {url}: {getattr(e, 'reason', e)}"
        if "pdf" in ct.lower() or raw[:5] == b"%PDF-":
            if not _has("pdftotext"):
                return "error: URL is a PDF and `pdftotext` is not installed (brew install poppler)."
            tmp = self.store / "dl.pdf"
            tmp.write_bytes(raw)
            r = subprocess.run(["pdftotext", "-layout", str(tmp), "-"], capture_output=True, text=True, timeout=120)
            text = r.stdout
        else:
            body = decode_body(raw, ct)
            text = html_to_text(body) if ("html" in ct.lower() or "<html" in body[:2000].lower() or "<body" in body[:5000].lower()) else body
        text = text.strip()
        if not text:
            return f"error: {final} returned no readable text (JavaScript-only page?). Try another source."
        return f"[fetched {final}]\n{text}"


# ---------------------------------------------------------------- delegation (cloud subagent)

SUB_TOOLS = ["web_search", "fetch", "read", "grep", "ls"]  # read-only on purpose
SUB_STEPS = 10
SUB_RESULT_CHARS = 12000  # per tool result; the subagent has a big context
SUB_TOTAL_CHARS = 150_000
SUB_SYSTEM = """You are a research subagent working for another AI (the lead). Working dir for files: {cwd}.
Do the task with your tools: search, then fetch the best primary sources (snippets are not enough); read files when asked.
Use at most {steps} tool calls. Then reply with a REPORT (no tool call), under 300 words:
- Findings: bullets with exact figures, names, dates, quotes; each with its source (URL or file path).
- Gaps: what you could not find or verify.
Facts only. No advice, no opinions, no disclaimers — the lead makes the judgment."""
REFUSAL = re.compile(
    r"\b(I can(?:no|')t (?:help|assist|provide|do|comply)|I(?:'m| am) (?:not able|unable) to|I won't|"
    r"cannot (?:help|assist|comply) with|against (?:my|the) (?:policy|policies|guidelines)|not appropriate for me)",
    re.I)


def delegate_task(task: str, tools: "Tools", log) -> str:
    """Run a DeepSeek tool-loop on `task`; return a short report, or a 'declined' note the lead can act on."""
    if not COMP_KEY:
        return "[delegate unavailable: no DEEPSEEK_API_KEY] Do it yourself with web_search/fetch/read."
    msgs = [{"role": "system", "content": SUB_SYSTEM.format(cwd=tools.cwd, steps=SUB_STEPS)},
            {"role": "user", "content": str(task)}]
    sub_schemas = [s for s in schemas_for(SUB_TOOLS)]
    sources, t0 = set(), time.time()

    def ask(with_tools: bool) -> dict:
        body = {"model": COMP_MODEL, "messages": msgs, "max_tokens": 1200, "temperature": 0.2,
                "thinking": {"type": "disabled"}}
        if with_tools:
            body["tools"] = sub_schemas
        return post_chat(COMP_URL, COMP_KEY, body, timeout=180)["choices"][0]

    try:
        for step in range(SUB_STEPS + 1):
            ch = ask(with_tools=step < SUB_STEPS)
            if ch.get("finish_reason") == "content_filter":
                return "[delegate declined: provider content filter] Do this part yourself with your own tools."
            msg = ch["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                report = (msg.get("content") or "").strip()
                if not report:
                    return "[delegate returned nothing] Do this part yourself with your own tools."
                if REFUSAL.search(report[:400]):
                    log(f"    [subagent refused: {report[:80]!r}]")
                    return f"[delegate declined: {report[:160]}] Do this part yourself with your own tools."
                log(f"    [subagent done: {step} tool steps, {time.time() - t0:.0f}s]")
                return (f"[subagent report — {step} tool steps, {len(sources)} sources; verify anything critical]\n"
                        + clip(report, INLINE_CHARS - 200))
            msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls[:MAX_CALLS_PER_STEP]:
                fn = c.get("function") or {}
                name = fn.get("name", "?")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = (tools.run(name, args) if name in SUB_TOOLS and isinstance(args, dict)
                       else f"error: tool {name!r} not available to subagent")
                if name in ("fetch", "read") and not out.startswith("error"):
                    sources.add(args.get("url") or args.get("path"))
                log(f"    ⤷ sub {name} {json.dumps(args, ensure_ascii=False)[:90]}")
                msgs.append({"role": "tool", "tool_call_id": c.get("id") or name, "content": clip(out, SUB_RESULT_CHARS)})
            for c in calls[MAX_CALLS_PER_STEP:]:  # every tool_call id needs a reply
                msgs.append({"role": "tool", "tool_call_id": c.get("id") or "x", "content": "skipped: too many calls"})
            while sum(len(m.get("content") or "") for m in msgs) > SUB_TOTAL_CHARS:
                old = next((m for m in msgs[2:] if m["role"] == "tool" and len(m["content"]) > 600), None)
                if old is None:
                    break
                old["content"] = old["content"][:500] + "\n…[trimmed]"
        return "[delegate gave no report] Do this part yourself with your own tools."
    except HTTPFail as e:
        txt = str(e)
        if "Content Exists Risk" in txt or "content" in txt.lower() and "risk" in txt.lower():
            return "[delegate declined: provider moderation] Do this part yourself with your own tools."
        return f"[delegate unavailable: {txt[:150]}] Do this part yourself with your own tools."
    except Exception as e:  # noqa: BLE001
        return f"[delegate failed: {type(e).__name__}: {str(e)[:150]}] Do this part yourself with your own tools."


def parse_xml_calls(text: str) -> list:
    """Fallback for Qwen's <tool_call><function=x><parameter=k>v</parameter> format in content."""
    calls = []
    for m in re.finditer(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", text, re.S):
        args = {k: v.strip("\n") for k, v in re.findall(r"<parameter=([\w.-]+)>(.*?)</parameter>", m.group(2), re.S)}
        calls.append({"id": uuid.uuid4().hex[:12], "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args)}})
    return calls


def clip(s: str, n: int, note: str = "…[clipped]…") -> str:
    return s if len(s) <= n else s[: n * 2 // 3] + f"\n{note}\n" + s[-(n // 3):]


# ---------------------------------------------------------------- agent


class Agent:
    def __init__(self, cwd: Path, budget: int, max_steps: int, think: bool, mode: str = "advisor",
                 stream: bool = True, quiet: bool = False):
        self.cwd, self.budget, self.max_steps, self.think = cwd, budget, max_steps, think
        self.mode, self.stream, self.quiet = mode, stream, quiet
        self.store = cwd / ".lowctx"
        self.store.mkdir(exist_ok=True)
        self.tools = Tools(cwd, self.store)
        self.tools.log = self.log
        self.model = LOCAL_MODEL
        self.comp = Compressor(self.log, self.get_model)
        self.goal = ""  # current user message (used to focus digests)
        self.notes = ""
        self.live: list[dict] = []
        self.chars_per_token = 3.2
        self.n_out = 0
        self.streamed = False
        self.stats = {"steps": 0, "compactions": 0, "digests": 0, "prompt_tokens": [], "secs": 0.0, "turns": 0}
        self.transcript = open(self.store / "transcript.jsonl", "a")

    # -- io
    def log(self, s: str) -> None:
        if not self.quiet:
            print(s, flush=True)

    def record(self, kind: str, **kw) -> None:
        self.transcript.write(json.dumps({"t": time.time(), "kind": kind, **kw}, ensure_ascii=False) + "\n")
        self.transcript.flush()

    def get_model(self) -> str:
        if self.model:
            return self.model
        try:
            req = urllib.request.Request(LOCAL_URL.rstrip("/") + "/models",
                                         headers={"Authorization": f"Bearer {LOCAL_KEY}"} if LOCAL_KEY else {})
            ids = [m["id"] for m in json.loads(urllib.request.urlopen(req, timeout=10).read())["data"]]
            # mlx_vlm lists cached HF models, not the loaded one, and asking for another id makes it
            # load that model. Only take a listed id when it is the server's single model and not mlx_vlm.
            self.model = ids[0] if len(ids) == 1 and not Path(DEFAULT_MODEL).exists() else DEFAULT_MODEL
        except Exception:  # noqa: BLE001
            self.model = DEFAULT_MODEL
        return self.model

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def reset(self) -> None:
        self.goal, self.notes, self.live = "", "", []

    # -- prompt assembly
    def system(self) -> dict:
        s = (SYSTEM_ADVISOR if self.mode == "advisor" else SYSTEM_CODE).format(cwd=self.cwd)
        if self.notes:
            s += f"\n\n## Notes (earlier conversation/work, summarized)\n{self.notes}"
        return {"role": "system", "content": s}

    def messages(self) -> list:
        return [self.system(), *self.live]

    def est(self, msgs: list) -> int:
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs) + len(json.dumps(schemas(self.mode)))
        return int(chars / self.chars_per_token)

    def sanitize(self) -> None:
        """Keep the live list valid for strict chat templates: starts with a user turn, tool results
        always match the assistant tool_calls just before them, no dangling calls."""
        out: list[dict] = []
        for m in self.live:
            if m["role"] == "tool":
                prev = next((x for x in reversed(out) if x["role"] != "tool"), None)
                ids = {c["id"] for c in (prev or {}).get("tool_calls") or []}
                if m.get("tool_call_id") in ids:
                    out.append(m)
                continue
            out.append(m)
        fixed: list[dict] = []
        for i, m in enumerate(out):
            fixed.append(m)
            if m["role"] == "assistant" and m.get("tool_calls"):
                have = set()
                j = i + 1
                while j < len(out) and out[j]["role"] == "tool":
                    have.add(out[j].get("tool_call_id"))
                    j += 1
                for c in m["tool_calls"]:
                    if c["id"] not in have:
                        fixed.append({"role": "tool", "tool_call_id": c["id"], "name": c["function"]["name"],
                                      "content": "(no result)"})
        while fixed and fixed[0]["role"] != "user":
            fixed.pop(0)
        self.live = fixed

    # -- compaction
    def fold(self, msgs: list) -> None:
        """Merge messages into the notes via the compressor."""
        events = []
        for m in msgs:
            if m["role"] == "assistant":
                for c in m.get("tool_calls") or []:
                    events.append(f"CALL {c['function']['name']} {c['function']['arguments'][:600]}")
                if m.get("content"):
                    events.append(f"ADVISOR: {m['content'][:2500]}")
            elif m["role"] == "tool":
                events.append(f"RESULT: {m['content'][:1800]}")
            elif m["role"] == "user":
                events.append(f"USER: {m['content'][:2500]}")
        if not events:
            return
        words = max(200, min(450, self.budget // 10))
        tmpl = NOTES_CHAT if self.mode == "advisor" else NOTES_CODE
        prompt = (tmpl.format(words=words) + f"\n\nOLD NOTES:\n{self.notes or '(none)'}\n\nNEW EVENTS:\n" + "\n".join(events))
        try:
            notes = self.comp.run(prompt, words, goal=self.goal)
        except Exception as e:  # noqa: BLE001
            self.log(f"  [compaction failed: {str(e)[:100]}; keeping crude notes]")
            notes = ""
        if not notes:  # crude fallback so nothing is silently lost
            notes = (self.notes + "\n" + "\n".join(e[:300] for e in events))[-words * 6:]
        self.notes = notes[: words * 8]
        self.stats["compactions"] += 1
        self.record("notes", notes=self.notes)

    def compact(self, force: bool = False) -> None:
        self.sanitize()
        if not force and self.est(self.messages()) <= self.budget:
            return
        users = [i for i, m in enumerate(self.live) if m["role"] == "user"]
        cur = users[-1] if users else 0
        reserve = int(max(200, min(450, self.budget // 10)) * 1.5)  # tokens a fresh notes block may add

        def groups(lst):  # assistant message + its tool results
            out: list[list[dict]] = []
            for m in lst:
                if m["role"] == "tool" and out:
                    out[-1].append(m)
                else:
                    out.append([m])
            return out

        def fits(lst):
            return self.est([self.system(), *lst]) + (0 if self.notes else reserve) <= self.budget * 0.8  # headroom, so we don't re-fold every step

        prev, turn = self.live[:cur], self.live[cur:]
        tg = groups(turn[1:])  # exchanges after the user message
        options = []  # (fold these, keep these)
        pu = [i for i, m in enumerate(prev) if m["role"] == "user"]
        if len(pu) >= 2 and not force:
            options.append((prev[: pu[-1]], prev[pu[-1]:] + turn))
        if prev:
            options.append((prev, turn))
        if len(tg) > KEEP_RECENT:
            options.append((prev + [m for g in tg[:-KEEP_RECENT] for m in g],
                            [turn[0]] + [m for g in tg[-KEEP_RECENT:] for m in g]))
        if force or not options or not fits(options[-1][1]):
            n = 1 if force else KEEP_RECENT
            options.append((prev + [m for g in tg[:-n] for m in g] if len(tg) > n else prev,
                            [turn[0]] + [m for g in (tg[-n:] if tg else []) for m in g]))
        chosen = next(((f, k) for f, k in options if not force and fits(k)), options[-1])
        fold_msgs, keep = chosen
        fold_msgs = [m for m in fold_msgs]
        if turn and turn[0] not in keep and turn[0] in fold_msgs:
            keep = [turn[0]] + keep
            fold_msgs.remove(turn[0])
        if fold_msgs:
            self.fold(fold_msgs)
        self.live = keep
        # last resort: shrink oversized tool results / messages in place
        limit = 1200 if force else 2400
        for m in self.live:
            if m["role"] == "tool" and len(m["content"]) > limit:
                m["content"] = clip(m["content"], limit, "…[trimmed; re-run tool if needed]…")
        if not fits(self.live):
            for m in self.live:
                if m["role"] == "tool" and len(m["content"]) > 800:
                    m["content"] = clip(m["content"], 800, "…[trimmed]…")
        self.sanitize()

    # -- results shaping
    def shape_result(self, name: str, args: dict, out: str) -> str:
        """Keep tool results small; save full output and give the model a digest."""
        limit = 4200 if name == "read" else INLINE_CHARS
        if len(out) <= limit or name in ("edit", "write", "delegate"):
            return out
        self.n_out += 1
        f = self.store / (f"page-{self.n_out}.txt" if name == "fetch" else f"out-{self.n_out}.txt")
        f.write_text(out)
        rel = f.relative_to(self.cwd)
        if name == "read":
            return clip(out, limit)
        call = f"{name} {json.dumps(args)[:300]}"
        try:
            digest = self.comp.run(DIGEST_PROMPT.format(goal=self.goal[:1200], call=call, words=220) +
                                   "\n\nOUTPUT:\n" + out, 220, goal=self.goal)
        except Exception as e:  # noqa: BLE001
            digest = "(digest failed: %s)\n%s" % (str(e)[:80], focus(out, self.goal, 1200))
        self.stats["digests"] += 1
        head = out.splitlines()[0] if name in ("bash", "fetch") else ""
        return (f"{head}\n[digest of {len(out)} chars; full text saved to {rel} — use grep on that file for specific terms; do not page through it]\n{digest}")[:MSG_CAP]

    def prepare_user(self, text: str) -> str:
        if len(text) / self.chars_per_token > self.budget * 0.3:
            self.n_out += 1
            f = self.store / f"input-{int(time.time())}-{self.n_out}.txt"
            f.write_text(text)
            self.log(f"  [large message ({len(text)} chars) saved to {f.relative_to(self.cwd)}; briefing it]")
            try:
                brief = self.comp.run(BRIEF_PROMPT.format(words=300, text=text), 300, goal=text[:500])
            except Exception as e:  # noqa: BLE001
                brief = focus(text, text[:300], 1500)
                self.log(f"  [briefing failed: {str(e)[:80]}; using excerpt]")
            text = (f"{brief}\n\n(The user pasted {len(text)} chars; condensed above. Full original saved at "
                    f"{f.relative_to(self.cwd)} — read it with start= if details are missing.)")
        return text

    # -- model call
    def call_local(self, tools: bool = True, nudge: str | None = None):
        """One model call with recovery: on a server error, force-compact and retry once."""
        for attempt in (0, 1):
            msgs = self.messages()
            if nudge:
                msgs = msgs + [{"role": "user", "content": nudge}]
            body = {"model": self.get_model(), "messages": msgs,
                    "max_tokens": 2048 if self.think else (1200 if self.mode == "advisor" else 1024),
                    "temperature": 0.3 if self.mode == "advisor" else 0.2}
            if tools:
                body["tools"] = schemas(self.mode)
            if THINK_FIELD:
                body["enable_thinking"] = self.think
            t = time.time()
            streamer = Streamer() if (self.stream and not self.quiet) else None
            try:
                if self.stream:
                    d = stream_chat(LOCAL_URL, LOCAL_KEY, body, streamer.feed if streamer else (lambda s: None))
                else:
                    d = post_chat(LOCAL_URL, LOCAL_KEY, body)
            except HTTPFail as e:
                if attempt == 0 and e.code in (400, 413, 422, 500, 502, 503):
                    self.log(f"  [server error {e.code}: {str(e)[:120]}; force-compacting and retrying once]")
                    self.compact(force=True)
                    continue
                raise
            break
        dt = time.time() - t
        u = d.get("usage") or {}
        pt = u.get("prompt_tokens")
        if pt:
            chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs) + (len(json.dumps(schemas(self.mode))) if tools else 0)
            self.chars_per_token = max(1.5, 0.7 * self.chars_per_token + 0.3 * (chars / pt))
            self.stats["prompt_tokens"].append(pt)
            self.turn_pts.append(pt)
        self.stats["secs"] += dt
        self.stats["steps"] += 1
        if streamer and streamer.printed:
            print()
            self.streamed = True
        self.log(f"  · step {self.stats['steps']}: prompt={pt} tok, out={u.get('completion_tokens')} tok, {dt:.0f}s")
        fr = d["choices"][0].get("finish_reason")
        return d["choices"][0]["message"], fr

    def run(self, request: str) -> str:
        """One user turn. Returns the final answer (already printed live if self.streamed)."""
        self.turn_pts, self.streamed = [], False
        t0, steps0, comp0 = time.time(), self.stats["steps"], self.stats["compactions"]
        text = self.prepare_user(request)
        self.goal = text
        self.sanitize()
        self.live.append({"role": "user", "content": text})
        start = len(self.live) - 1
        self.record("user", text=text)
        seen: dict = {}
        trips = 0
        final = None
        try:
            for step in range(self.max_steps):
                self.compact()
                nudge = None
                if trips >= 2:
                    nudge = "You are repeating yourself. Stop calling tools and answer now with what you have."
                msg, fr = self.call_local(tools=nudge is None)
                content = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
                calls = msg.get("tool_calls") or []
                if nudge is None and not calls and "<function=" in content:
                    calls = parse_xml_calls(content)
                if "<tool_call>" in content or "<function=" in content:
                    content = re.split(r"<tool_call>|<function=", content)[0].strip()
                if nudge is not None:
                    calls = []
                if not calls:
                    if not content:
                        if step < self.max_steps - 1 and not seen.get("_empty"):
                            seen["_empty"] = 1
                            self.log("  [empty reply; retrying]")
                            continue
                        content = "(the model returned an empty reply)"
                    if fr == "length":
                        content += "\n[reply cut off at the token limit; ask me to continue]"
                    final = content
                    break
                calls = calls[:MAX_CALLS_PER_STEP]
                for c in calls:
                    c.setdefault("id", uuid.uuid4().hex[:12])
                    c["id"] = c["id"] or uuid.uuid4().hex[:12]
                if content and not self.streamed:
                    self.log(f"  {content[:200]}")
                self.live.append({"role": "assistant", "content": content, "tool_calls": calls})
                for c in calls:
                    fn = c.get("function") or {}
                    name = fn.get("name", "?")
                    raw = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                        if not isinstance(args, dict):
                            raise ValueError("not an object")
                    except (json.JSONDecodeError, ValueError):
                        args, out = {}, f"error: arguments were not a valid JSON object: {str(raw)[:200]}"
                    else:
                        sig = (name, json.dumps(args, sort_keys=True))
                        seen[sig] = seen.get(sig, 0) + 1
                        if seen[sig] >= 3:
                            trips += 1
                            out = f"error: you already made this exact call {seen[sig] - 1} times with the same result. Do not repeat it; use what you have and answer."
                        else:
                            out = self.tools.run(name, args)
                            if seen[sig] == 2:
                                out += "\n[note: you already made this exact call. Use the result and move on.]"
                    self.log(f"  → {name} {json.dumps(args, ensure_ascii=False)[:110]}")
                    self.record("tool", name=name, args=args, out=out[:20000])
                    self.live.append({"role": "tool", "tool_call_id": c["id"], "name": name,
                                      "content": self.shape_result(name, args, out)[:MSG_CAP]})
                if step + 1 >= max(6, self.max_steps * 0.4):
                    self.live[-1]["content"] += f"\n[step {step + 1}/{self.max_steps}: you have enough; make at most one more call, then answer]"
            if final is None:  # step limit: force an answer from what we have
                self.log("  [step limit reached; forcing a final answer]")
                self.compact()
                msg, fr = self.call_local(tools=False, nudge="Step limit reached. Answer now with what you have; say what is missing.")
                final = re.split(r"<tool_call>|<function=", re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S))[0].strip() or "(no answer)"
        except Exception:
            self.live = self.live[:start]  # roll back the failed turn; state stays valid
            self.sanitize()
            raise
        self.live.append({"role": "assistant", "content": final[:MSG_CAP]})
        for m in self.live[start:]:  # finished turn: clip bulky tool results
            if m["role"] == "tool" and len(m["content"]) > OLD_TOOL_CAP:
                m["content"] = clip(m["content"], OLD_TOOL_CAP, "…[older result clipped]…")
        self.record("final", text=final)
        self.stats["turns"] += 1
        self.last_turn = {"steps": self.stats["steps"] - steps0, "secs": time.time() - t0,
                          "max_prompt": max(self.turn_pts or [0]), "compactions": self.stats["compactions"] - comp0}
        return final

    def summary(self) -> str:
        lt = getattr(self, "last_turn", None) or {}
        return (f"[turn: {lt.get('steps')} steps, {lt.get('secs', 0):.0f}s, max prompt {lt.get('max_prompt')} tok, "
                f"{lt.get('compactions')} compactions | session: {self.stats['turns']} turns, "
                f"{self.stats['compactions']} compactions, {self.stats['digests']} digests, "
                f"max prompt {max(self.stats['prompt_tokens'] or [0])} tok, compressor={self.comp.name}, mode={self.mode}]")


# ---------------------------------------------------------------- cli


HELP = ("commands: /new  /notes  /mode code|advisor  /help  /quit\n"
        'multi-line: wrap in """ ... """, end a line with \\, or just paste (pasted lines are joined)\n')


def read_input(prompt: str = "you> ") -> str:
    tty = sys.stdin.isatty()
    first = input(prompt)
    lines = [first]
    if first.strip().startswith('"""'):
        body = first.strip()[3:]
        if body.endswith('"""') and len(body) >= 3:
            return body[:-3].strip()
        lines = [body]
        while True:
            l = input("...  ")
            if l.rstrip().endswith('"""'):
                lines.append(l.rstrip()[:-3])
                break
            lines.append(l)
        return "\n".join(lines).strip()
    while lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]
        lines.append(input("...  "))
    if tty:  # a multi-line paste arrives as several lines at once
        try:
            while select.select([sys.stdin], [], [], 0.08)[0]:
                l = sys.stdin.readline()
                if not l:
                    break
                lines.append(l.rstrip("\n"))
        except (OSError, ValueError):
            pass
    return "\n".join(lines).strip()


def main(argv: list | None = None, header=None, prog: str | None = None) -> None:
    """CLI loop. `header(agent)` replaces the plain startup line (porthole draws its banner there)."""
    global LOCAL_URL, LOCAL_MODEL
    ap = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="*", help="optional first message")
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("--mode", choices=["advisor", "code"], default=os.environ.get("LOWCTX_MODE", "advisor"))
    ap.add_argument("--budget", type=int, default=int(os.environ.get("LOWCTX_BUDGET", "5000")),
                    help="target prompt tokens for the local model (default 5000)")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("LOWCTX_STEPS", "20")), help="max tool steps per turn")
    ap.add_argument("--url", help="OpenAI-compatible base URL (env LOWCTX_URL)")
    ap.add_argument("--model", help="model id (env LOWCTX_MODEL; default autodetect)")
    ap.add_argument("--think", action="store_true", help="enable model thinking (slower)")
    ap.add_argument("--no-stream", action="store_true", help="do not stream the answer")
    ap.add_argument("--once", action="store_true", help="answer the given task and exit")
    a = ap.parse_args(argv)
    if a.url:
        LOCAL_URL = a.url
    if a.model:
        LOCAL_MODEL = a.model

    agent = Agent(Path(a.cwd).resolve(), a.budget, a.steps, a.think, a.mode, stream=not a.no_stream)
    if header:
        header(agent)
    else:
        print(f"lowctx  mode={agent.mode}  cwd={agent.cwd}  budget={a.budget} tok  compressor={agent.comp.name}  "
              f"model={agent.get_model().split('/')[-2:] if '/' in agent.get_model() else agent.get_model()}")
        print(HELP)
    task = " ".join(a.task).strip()
    while True:
        if not task:
            try:
                task = read_input()
            except (EOFError, KeyboardInterrupt):
                print()
                return
        low = task.lower()
        if low in ("/quit", "/exit", "/q"):
            return
        if low == "/new":
            agent.reset()
            print("new conversation (notes cleared)")
        elif low == "/notes":
            print(agent.notes or "(no notes yet)")
        elif low == "/help":
            print(HELP)
        elif low.startswith("/mode"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in ("code", "advisor"):
                agent.set_mode(parts[1])
                print(f"mode: {agent.mode}")
            else:
                print(f"mode: {agent.mode}  (usage: /mode code|advisor)")
        elif task.startswith("/") and " " not in task and len(task) < 20:
            print("unknown command; /help")
        elif task:
            try:
                ans = agent.run(task)
                if not agent.streamed:
                    print("\n" + ans)
                print(agent.summary() + "\n")
            except KeyboardInterrupt:
                agent.sanitize()
                print("\n(interrupted)")
            except Exception as e:  # noqa: BLE001
                print(f"\n[error: {e}] — the turn was rolled back; try again or /new\n")
        task = ""
        if a.once:
            return


if __name__ == "__main__":
    main()
