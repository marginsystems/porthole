import json, sys, time, urllib.request, random
API = "http://127.0.0.1:8080/v1/chat/completions"
import os, pathlib
MODEL = str(pathlib.Path(__file__).resolve().parents[1] / "Qwen3.8-27B-Uncensored-MLX" / os.environ.get("QWEN_QUANT", "4-bit"))
def call(messages, **kw):
    body = {"model": MODEL, "messages": messages, "max_tokens": kw.pop("max_tokens", 300), "temperature": 0.2, **kw}
    t = time.time()
    req = urllib.request.Request(API, json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return d, time.time() - t
mode = sys.argv[1]
if mode == "tools":
    tools = [{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    for et in (None, False):
        kw = {"tools": tools}
        if et is not None: kw["enable_thinking"] = et
        d, dt = call([{"role":"user","content":"Read the file setup.py please."}], **kw)
        m = d["choices"][0]["message"]
        print(f"enable_thinking={et} {dt:.1f}s usage={d.get('usage')}")
        print(" content:", repr((m.get("content") or "")[:400]))
        print(" reasoning:", repr((m.get("reasoning_content") or m.get("reasoning") or "")[:200]))
        print(" tool_calls:", m.get("tool_calls"))
elif mode == "needle":
    random.seed(1)
    words = "the a system file module function returns value config server cache index request user data token build test".split()
    for n in [int(x) for x in sys.argv[2].split(",")]:
        filler = [" ".join(random.choice(words) for _ in range(12)) + "." for _ in range(n // 14)]
        filler.insert(len(filler) // 10, "IMPORTANT: the secret deploy code is PELICAN-7741.")
        msg = "\n".join(filler) + "\n\nWhat is the secret deploy code? Answer with just the code."
        d, dt = call([{"role":"user","content":msg}], max_tokens=40, enable_thinking=False)
        u = d.get("usage", {})
        print(f"~{n} words  prompt_tokens={u.get('prompt_tokens')}  {dt:.1f}s  -> {d['choices'][0]['message'].get('content')!r}", flush=True)
if mode == "cache":
    base = [{"role":"system","content":"You are helpful. " + ("Background: the project uses python and pytest. " * 150)}]
    for q in ["Say hi.", "Say bye."]:
        d, dt = call(base + [{"role":"user","content":q}], max_tokens=10, enable_thinking=False)
        print(f"prefix-cache {dt:.1f}s usage={d.get('usage')}")
    d, dt = call([{"role":"user","content":"Write a 250-word explanation of Python generators."}], max_tokens=350, enable_thinking=False)
    c = d["usage"]["completion_tokens"]; print(f"decode: {c} tokens in {dt:.1f}s = {c/dt:.1f} tok/s")
