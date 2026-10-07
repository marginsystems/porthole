# porthole

```
   ▄▄█▀▀▀█▄▄
 ▄█ ●     ● █▄    ___  ___  ___ _____ _  _  ___  _    ___
█▀  ▗▄▄▄▄▄▖  ▀█  | _ \/ _ \| _ \_   _| || |/ _ \| |  | __|
█   ▐░░▒▓▒▌   █  |  _/ (_) |   / | | | __ | (_) | |__| _|
█▄  ▝▀▀▀▀▀▘  ▄█  |_|  \___/|_|_\ |_| |_||_|\___/|____|___|
 ▀█ ●     ● █▀   uncensored local minds · tiny context, long reach
   ▀▀█▄▄▄█▀▀
```

**A small window onto a big, unfiltered mind.** porthole is an agent harness for running an uncensored
27B Qwen *locally* on a Mac. It gives the model a small context window to work in, and cloud subagents
for the legwork, so it can read your files, search the web, run commands and fix code across long sessions
without the context falling apart.

Out of the box it's a blunt advisor that tells you what's wrong with your business plan, your code or your
pitch, without flattery or hedging. It runs on your machine.

## Why this exists

Quantized local models run on a laptop, but they degrade quickly once the prompt gets long. Every agent step
re-reads the whole prompt with no prefix cache, at roughly 100 tokens/s of prefill, so a 16k-token prompt costs
about 4 minutes *per step*. A normal agent loop of tool calls, file dumps and long chat history kills them.

porthole keeps every step's prompt at about **12k tokens** or less no matter how long the session runs:

```
                                ┌──────────── Qwen, local (the orchestrator) ────────────┐
  you ──▶ request ──▶ prompt ≤~12k tok: system · request · working notes · last 2 steps   │──▶ answer
                                └───┬──────────────────────────┬─────────────────────────┘
                                    │ ask("what's the KV cap   │ ask("cheapest Trello tier,
                                    │  in start-server.sh?")   │  with URL")      ← fired together,
                                    ▼                          ▼                   run in parallel
                          ┌──────────────────┐       ┌──────────────────┐
                          │ DeepSeek worker  │       │ DeepSeek worker  │   read-only tools:
                          │ reads whole files│       │ searches, fetches│   read · grep · ls ·
                          └────────┬─────────┘       └────────┬─────────┘   web_search · fetch
                                   └── ANSWER + EVIDENCE (file:line / URL), ≤200 words ──▶ back to Qwen
```

* **Qwen holds the big picture.** It understands your request, plans, judges, edits and writes the answer.
* **Subagents answer narrow questions.** Instead of paging through a file to learn one fact, Qwen calls
  `ask(question, hints)`: *"what is the monthly churn in revenue.csv?"*, *"where is `retry_limit` set and to
  what?"*, *"cheapest paid ClickUp tier, with URL"*. Each ask is an independent DeepSeek worker with a large
  context and read-only tools. It reads whole files, greps and browses, then returns only
  `ANSWER / EVIDENCE / UNSURE`. All asks in one step run in parallel, on a live board in the CLI.
* **Workers never see your request.** They get only Qwen's question. Understanding the task is Qwen's job;
  DeepSeek just does lookups.
* **If a worker refuses** (refusal text, content filter, moderation error) or fails, Qwen gets
  `[ask declined …] Find it yourself` and uses its own read/grep/search. Nothing stops.
* **The compressor keys on Qwen's calls too.** Big tool output is reduced to what *that call* was after (the
  call's `purpose`, or Qwen's own words before it), and old steps are folded into a factual log of calls and
  results. Without a key, the local model compresses its own context: slower, but fully offline.

## Requirements

* Apple Silicon Mac (MLX). 32 GB RAM or more recommended for the 4-bit model (it uses about 16 GB plus a
  working margin); 64 GB+ for 6/8-bit.
* Python 3.10+, ~16 GB free disk for the 4-bit weights.
* Optional: a [DeepSeek API key](https://platform.deepseek.com) for subagents and fast compression;
  `brew install ripgrep poppler` for faster grep and PDF reading.

## Quickstart

```bash
git clone https://github.com/marginsystems/porthole.git
cd porthole
./install.sh            # venv + mlx-vlm, links `porthole` into ~/.local/bin, creates .env
porthole pull           # downloads the 4-bit uncensored Qwen (~16 GB, resumable)
$EDITOR .env            # optional: DEEPSEEK_API_KEY=sk-...
porthole doctor         # checks everything
porthole                # starts the model server if needed, opens a chat in the current folder
```

## Getting the uncensored Qwen

porthole defaults to **[orcarouter/Qwen3.8-27B-Uncensored-MLX](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX)**
(Apache-2.0), an uncensored fine-tune of Qwen 3.8 27B converted to MLX. It is a hybrid linear-attention model:
only 16 of its 64 layers keep a KV cache, so long context is cheap in memory. The bottleneck is compute.

| build | size | fits on | notes |
|---|---|---|---|
| `2-bit` | 9.4 GB | 16–24 GB Macs | noticeably dumber; last resort |
| **`4-bit`** | **16.1 GB** | **32 GB+** | **default.** Best quality per GB; passes the coding tests |
| `6-bit` | 22.8 GB | 48 GB+ (tight) | |
| `8-bit` | ~29 GB | 64 GB+ | made a 48 GB Mac swap hard next to a browser |

```bash
porthole pull            # 4-bit (default)
porthole pull 6-bit      # another build; then run with QWEN_QUANT=6-bit
```

Or download it yourself:
`hf download orcarouter/Qwen3.8-27B-Uncensored-MLX --include "4-bit/*" --local-dir Qwen3.8-27B-Uncensored-MLX`.
If you already have the 8-bit build, `requant.py` converts it to 4-bit one shard at a time (peak memory a few GB):
`.venv/bin/python requant.py Qwen3.8-27B-Uncensored-MLX/8-bit Qwen3.8-27B-Uncensored-MLX/4-bit`.

"Uncensored" means the model was tuned not to refuse. That's the point here: candid answers about your own
work. What you do with it is on you. Read the model card.

## Commands

| command | what it does |
|---|---|
| `porthole` | chat in the current folder (advisor mode); starts the model server if it's down |
| `porthole chat --mode code` | coding-agent mode: ask/ls/read/grep/edit/write/bash |
| `porthole run "task"` | one-shot: do the task in this folder, print the result, exit (code mode) |
| `porthole run --mode advisor "question"` | one-shot opinion |
| `porthole up` / `porthole down` | start / stop the model server. **down frees ~16 GB**; do it when you're done |
| `porthole status` | server, model, subagents, free memory |
| `porthole doctor` | checks the machine and the install |
| `porthole pull [quant]` | downloads weights from Hugging Face |

In chat: `/new` clears the conversation, `/notes` shows the compressed memory, `/mode code|advisor` switches
mode, `/quit` exits. Paste freely; multi-line pastes are joined, or wrap long text in `"""`. Huge pastes are
saved to `.lowctx/` and replaced by a brief, and the model can still read the original.

Scratch data (saved pages, big outputs, `transcript.jsonl`) goes to `./.lowctx/` in the folder you chat from.

## Use cases

* **Brutal business review.** `cd ~/my-startup && porthole`, then ask: *"Read everything here and tell me
  what's wrong with this business and what to do in the next 30 days."* In our test it read a plan, a revenue
  CSV and pitch notes, did the churn math itself and called out the pitch: *"The pitch notes call $1.2k→$2.9k
  'strong organic growth.' It's not."*
* **Pricing and competitor research.** *"We charge $9/seat. Check what Trello, Asana and ClickUp charge and
  tell me bluntly if we're mispriced."* The subagent fetches the official pricing pages (about 20s); Qwen weighs
  them against your own numbers.
* **Fix code until tests pass.** `porthole run "tests are failing; fix the bugs in src/, not the tests. Run: pytest -q"`.
  The bundled fixture (5 planted bugs) is fixed in 5–8 steps with prompts under 3.7k tokens.
* **Stress-test a document.** Point it at a contract, a PDF deck or an investor update: *"Where would a skeptical
  investor tear this apart?"*
* **Pre-mortems and red-teaming your own stuff.** Landing page copy, a hiring plan, a launch checklist:
  *"Assume this failed in 6 months. Why?"*
* **Offline thinking partner.** Leave `DEEPSEEK_API_KEY` empty and no model outside your Mac sees anything. Use it
  for strategy, journaling or anything you'd rather not send to a cloud model. In advisor mode, `web_search` and
  `fetch` still go online when Qwen uses them; switch to `/mode code` or just don't ask for research.

## Privacy

| stays on your Mac | goes to DeepSeek (only if a key is set) |
|---|---|
| the model and every prompt Qwen sees | each `ask` question and everything its worker reads or fetches, **including your local files** |
| file edits and bash commands Qwen runs itself | text the compressor summarizes: old turns of your chat, big tool output, long pastes |
| `.lowctx/` (saved pastes, outputs, transcript): workers can't open it | |

With a key set, Qwen sends most file lookups to workers, so assume the files it works with reach DeepSeek.
Workers can only read, grep and list inside the folder you launched porthole from. They can never open
credential-looking files (`.env*`, `*.pem`/`*.key`, `id_rsa*`, `~/.ssh`, `.aws`, `credentials*`, …), even if a
file or web page they read tells them to.
For sensitive folders, remove the key: the `ask` tool disappears and compression runs on the local model, so no
cloud model sees your files or chat. Web access is separate: in advisor mode `web_search` sends queries to
DuckDuckGo (Bing as fallback) and `fetch` downloads whatever URL Qwen picks, key or not. Code mode has no web tools.

## Configuration

`.env` in the repo root is loaded automatically; real environment variables win.

| env | default | |
|---|---|---|
| `DEEPSEEK_API_KEY` | none | enables subagents + fast compressor |
| `DEEPSEEK_MODEL` / `DEEPSEEK_URL` | `deepseek-flash` / `https://api.deepseek.com/v1` | the API accepts `deepseek-flash`, `deepseek-v4-pro` |
| `QWEN_QUANT` | `4-bit` | which build `porthole up` loads and the harness requests |
| `QWEN_KV` | `16384` | server context cap in tokens (prompts ≤~12k + up to 1.2k of answer + headroom) |
| `QWEN_PREFILL` | `512` | prefill chunk; larger chunks ran Metal out of memory on 16k prompts |
| `LOWCTX_URL` / `LOWCTX_MODEL` / `LOWCTX_KEY` | local mlx server | point porthole at **any OpenAI-compatible model** (LM Studio, llama.cpp, Ollama's `/v1`, vLLM…) |
| `LOWCTX_BUDGET` | `12000` | target prompt tokens per step (`--budget`). Lower = faster steps, more compaction |
| `LOWCTX_STEPS` | `20` | max tool steps per turn (`--steps`) |
| `LOWCTX_THINK_FIELD=0` | | don't send `enable_thinking` (for servers that reject it) |

The server runs at low CPU/IO priority (`taskpolicy -c utility`) so your other apps win when they compete.
mlx_vlm swaps models when a request names a different one, so keep `QWEN_QUANT` the same in every shell.

## Measured (M4 Max, 48 GB)

| | 8-bit | 4-bit |
|---|---|---|
| prefill, 4k-token prompt | ~40 s | ~21 s |
| decode | ~10 tok/s | ~17 tok/s |
| free RAM with server up | ~6% (swapping) | ~29% |
| coding fixture (5 bugs) | fixed, 6–8 steps | fixed, 5 steps |

| | time |
|---|---|
| compression step via DeepSeek | ~1.5–2.5 s |
| compression step via local Qwen | 1–2 min |
| research subagent (3 pricing pages) | ~20 s |
| 3 parallel `ask`s about this repo (budget, KV cap, env vars) | 5 s wall, all correct with file:line |
| advisor turn with one research subagent | 192 s, max prompt 3.4k tok (vs 267 s / 4.8k researching by hand) |

## Tests

```bash
.venv/bin/python bench/run_tests.py [t1] [biz] [research] [long] [paste]
```

Live tests against the running model, one request at a time. They take tens of minutes and keep the GPU busy.
Fixtures: `bench/t1` (a small package with 5 planted bugs) and `bench/biz` (a fictional SaaS with obvious
problems). `bench/probe.py tools|needle N,N|cache` probes tool calling, long-context recall and speed.

## Known limitations

* Local is slow: expect 20–60 s per agent step and ~17 tok/s output. Use `porthole down` when idle.
* Memory is everything. If other apps (VMs, games, DAWs, a heavy browser) already push the Mac into swap, a
  4k-token step can go from ~20 s to several minutes. `porthole up` warns when free RAM is below what the weights need.
* Notes are lossy. Exact figures from many turns back survive only if the compressor kept them
  (`.lowctx/transcript.jsonl` has everything).
* Subagent reports can contain mistakes (one ranked "$7 < $5" in a summary line); Qwen is told to verify
  anything critical, and so should you.
* JavaScript-rendered pages come back without their data; the subagent falls back to other sources.
* DuckDuckGo may rate-limit heavy use; the Bing fallback is regex-parsed and fragile.
* The local server is MLX-only (Apple Silicon). On other hardware, run any OpenAI-compatible server and set
  `LOWCTX_URL` / `LOWCTX_MODEL`.

## Layout

```
porthole          CLI (banner, up/down/status/doctor/pull, chat/run)
lowctx.py         the engine: context budgeting, compaction, tools, delegation, chat loop (stdlib only)
start-server.sh   mlx_vlm server with memory-safe defaults
install.sh        venv + mlx-vlm + PATH link
requant.py        8-bit → 4-bit re-quantizer (shard by shard)
bench/            live tests and fixtures
```

## License

MIT, see [LICENSE](LICENSE). The model weights are not part of this repo and carry their own license
(the default Qwen build is Apache-2.0; check the model card).
