#!/bin/bash
# porthole installer: venv + mlx-vlm, a `porthole` command on your PATH, and a .env to fill in.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
say() { printf "  \033[38;5;178m●\033[0m %s\n" "$*"; }
die() { printf "  \033[38;5;203m●\033[0m %s\n" "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "porthole's local server needs Apple Silicon macOS (MLX)."
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || die "need Python 3.10+ (brew install python)"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  say "creating virtualenv"
  "$PY" -m venv "$ROOT/.venv"
fi
say "installing mlx-vlm (MLX inference server)"
"$ROOT/.venv/bin/pip" install -q --upgrade pip
"$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"

[ -f "$ROOT/.env" ] || { cp "$ROOT/.env.example" "$ROOT/.env"; chmod 600 "$ROOT/.env"; say "created .env (add your DeepSeek key)"; }

BIN="${PORTHOLE_BIN:-$HOME/.local/bin}"
mkdir -p "$BIN"
ln -sf "$ROOT/porthole" "$BIN/porthole"
say "linked $BIN/porthole"
case ":$PATH:" in *":$BIN:"*) ;; *) say "add to your shell: export PATH=\"$BIN:\$PATH\"";; esac

say "next: porthole pull   (downloads ~16 GB of 4-bit Qwen weights)"
say "then: porthole        (starts the model and opens a chat)"
