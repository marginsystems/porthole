#!/usr/bin/env python3
"""Offline unit tests for lowctx (no model server, no network): python3 bench/test_unit.py"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lowctx  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name).resolve()
        self.agent = lowctx.Agent(self.cwd, 12000, 20, False, quiet=True)
        self.calls = []

        def fake_run(prompt, words, goal=""):
            self.calls.append((prompt, words))
            return f"BRIEF{len(self.calls)}"
        self.agent.comp.run = fake_run
        self.agent.comp.use_remote = True  # one brief call per paste; Paste tests switch it off for chunking

    def tearDown(self):
        self.agent.transcript.close()
        self.tmp.cleanup()


class WorkerAccess(Base):
    def test_saved_paste_readable_session_data_not(self):
        t, store = self.agent.tools, self.cwd / ".lowctx"
        (store / "input-1-1.txt").write_text(lowctx.PASTE_MARK + "alpha\nthe secret word is plum\n")
        (store / "input-7-7.txt").write_text("alpha\nno marker: not written by save_input\n")
        self.assertIn("off limits", lowctx.worker_blocked(t, "read", {"path": ".lowctx/input-7-7.txt"}))
        for f in ("session.json", "notes.md", "transcript.jsonl", "out-3.txt", "page-2.txt"):
            (store / f).write_text("x")
        self.assertIsNone(lowctx.worker_blocked(t, "read", {"path": ".lowctx/input-1-1.txt"}))
        self.assertIsNone(lowctx.worker_blocked(t, "grep", {"pattern": "plum", "path": ".lowctx/input-1-1.txt"}))
        for f in ("session.json", "notes.md", "transcript.jsonl", "out-3.txt", "page-2.txt", ""):
            self.assertIn("off limits", lowctx.worker_blocked(t, "read", {"path": f".lowctx/{f}"}) or "", f)
        self.assertIn("off limits", lowctx.worker_blocked(t, "read", {"path": ".lowctx/input-1-1.txt/../session.json"}))
        self.assertIn("off limits", lowctx.worker_blocked(t, "grep", {"pattern": "x", "path": ".lowctx"}))
        (self.cwd / ".env").write_text("KEY=1\n")  # a paste-shaped name linking elsewhere is judged by its target
        os.symlink(self.cwd / ".env", store / "input-9-9.txt")
        self.assertIn("off limits", lowctx.worker_blocked(t, "read", {"path": ".lowctx/input-9-9.txt"}))
        os.link(self.cwd / ".env", store / "input-8-8.txt")  # a hard link is .env itself under a paste name
        self.assertIn("off limits", lowctx.worker_blocked(t, "read", {"path": ".lowctx/input-8-8.txt"}))
        (self.cwd / ".env").unlink()  # ... and still is once the original name is gone (link count back to 1)
        self.assertIn("off limits", lowctx.worker_blocked(t, "read", {"path": ".lowctx/input-8-8.txt"}))
        os.link(store / "input-1-1.txt", store / "input-6-6.txt")  # a real paste linked twice is not a paste either
        self.assertIn("off limits", lowctx.worker_blocked(t, "read", {"path": ".lowctx/input-1-1.txt"}))
        (store / "input-6-6.txt").unlink()
        (store / "input-2-2.txt").mkdir()  # only files, never a directory to grep through
        self.assertIn("off limits", lowctx.worker_blocked(t, "grep", {"pattern": "x", "path": ".lowctx/input-2-2.txt"}))
        out = t.run("grep", {"pattern": "plum", "path": ".lowctx/input-1-1.txt"})
        self.assertIn("plum", out)
        self.assertIn("plum", t.read_full(".lowctx/input-1-1.txt"))

    def test_filter(self):
        out = lowctx.worker_filter(".lowctx/input-1-1.txt:3:see .env\n.lowctx/session.json:1:x\n.env:1:KEY=1\nok.py:2:y")
        self.assertEqual(out.splitlines(), [".lowctx/input-1-1.txt:3:see .env", "ok.py:2:y"])


class Paste(Base):
    def test_typed_question_kept_paste_briefed(self):
        paste = "\n".join(f"log line {i}: status ok" for i in range(800))
        msg = lowctx.UserText(f"why does this log say ok?\n{paste}\nand is it fine?")
        msg.pastes = (paste,)
        out = self.agent.prepare_user(msg)
        self.assertTrue(out.startswith("why does this log say ok?\n[pasted "))
        self.assertIn("and is it fine?", out)
        self.assertNotIn("log line 5:", out)
        self.assertIn("BRIEF1", out)
        saved = sorted((self.cwd / ".lowctx").glob("input-*.txt"))
        self.assertEqual([p.read_text() for p in saved], [lowctx.PASTE_MARK + paste])
        self.assertIn(f".lowctx/{saved[0].name}", out)
        self.assertIn("why does this log say ok?", self.calls[0][0])  # the typed words steer the brief
        self.assertEqual(len(self.calls), 1)

    def test_short_message_untouched(self):
        self.assertEqual(self.agent.prepare_user("hi"), "hi")
        self.assertEqual(self.calls, [])

    def test_no_paste_structure_falls_back(self):
        out = self.agent.prepare_user("x " * 20000)
        self.assertTrue(out.startswith("BRIEF1"))
        self.assertIn("Full original saved at .lowctx/input-", out)

    def test_huge_paste_local_compressor_chunks(self):
        self.agent.comp.use_remote = False
        paste = "\n".join(f"row {i} " + "z" * 90 for i in range(2000))  # ~200k chars: over 10 parts
        msg = lowctx.UserText("summarise this\n" + paste)
        msg.pastes = (paste,)
        out = self.agent.prepare_user(msg)
        self.assertEqual(len(self.calls), lowctx.BRIEF_MAX_PARTS + 1)  # 10 parts + 1 merge
        self.assertTrue(all(len(p) <= self.agent.comp.max_in for p, _ in self.calls))
        self.assertIn("part 1 of 10", self.calls[0][0])
        self.assertIn("BRIEF1", self.calls[-1][0])  # the merge sees the part briefs
        self.assertIn("BRIEF11", out)
        self.assertIn("only the first", out)
        self.assertEqual(self.calls[-1][1], 600)

    def test_brief_words_scale(self):
        self.assertEqual(lowctx.brief_words(5000), 300)
        self.assertEqual(lowctx.brief_words(12000), 300)
        self.assertEqual(lowctx.brief_words(10**6), 600)

    def test_line_editor_reports_pastes(self):
        ed = lowctx.LineEditor("> ")
        ed.buf = list("q: ") + [lowctx._Paste("a\nb  ")]
        r = ed.result()
        self.assertEqual(r, "q: a\nb")
        self.assertEqual(r.pastes, ("a\nb",))
        self.assertEqual(r.spans, ((3, 6),))

    def test_typed_copy_before_paste_is_left_alone(self):
        block = "\n".join(f"row {i}: value {i}" for i in range(700))
        ed = lowctx.LineEditor("> ")
        ed.buf = list(block + "\nsame again:\n") + [lowctx._Paste(block)] + list("\nwhich differs?")
        msg = ed.result()
        self.assertEqual(msg.spans[0][0], len(block) + len("\nsame again:\n"))
        out = self.agent.prepare_user(msg)
        self.assertTrue(out.startswith(block + "\nsame again:\n[pasted "))  # the typed copy stays, the paste goes
        self.assertEqual(out.count("row 5: value 5"), 1)
        self.assertIn("which differs?", out)

    def test_bad_extra_body_is_reported(self):
        import subprocess
        r = subprocess.run([sys.executable, "-c", "import lowctx; print(lowctx.HELPER_EXTRA_BODY, bool(lowctx.HELPER_EXTRA_ERR))"],
                           cwd=Path(lowctx.__file__).parent, capture_output=True, text=True,
                           env={**os.environ, "HELPER_EXTRA_BODY": "{'reasoning': 1}"})
        self.assertEqual(r.stdout.strip(), "{} True")
        self.assertIn("HELPER_EXTRA_BODY ignored", r.stderr)


class Session(Base):
    def test_mode_round_trip_and_notes_md(self):
        a = self.agent
        a.notes, a.live = "- user wants X", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        a.set_mode("code")
        a.save_session()
        self.assertEqual(json.loads(a.session_file.read_text())["mode"], "code")
        self.assertIn("- user wants X", a.notes_file.read_text())
        b = lowctx.Agent(self.cwd, 12000, 20, False, quiet=True)
        self.assertTrue(b.load_session())
        self.assertEqual((b.mode, b.notes), ("code", "- user wants X"))
        c = lowctx.Agent(self.cwd, 12000, 20, False, mode="advisor", quiet=True)
        c.load_session(restore_mode=False)
        self.assertEqual(c.mode, "advisor")
        c.reset()
        self.assertFalse(a.session_file.exists() or a.notes_file.exists())
        b.transcript.close()
        c.transcript.close()

    def test_mode_only_session_and_bad_mode(self):
        self.agent.set_mode("code")
        self.agent.save_session()  # what /mode does before any turn
        b = lowctx.Agent(self.cwd, 12000, 20, False, quiet=True)
        self.assertIsNone(b.load_session())
        self.assertEqual(b.mode, "code")
        data = json.loads(b.session_file.read_text())
        data["mode"] = ["evil"]
        b.session_file.write_text(json.dumps(data))
        b.set_mode("advisor")
        b.load_session()
        self.assertEqual(b.mode, "advisor")
        b.transcript.close()

    @unittest.skipIf(os.environ.get("LOWCTX_MODE", "advisor") != "advisor", "needs the advisor default")
    def test_main_restores_mode_unless_explicit(self):
        self.agent.set_mode("code")
        self.agent.save_session()
        seen, orig = [], lowctx.read_input

        def eof():
            raise EOFError
        lowctx.read_input = eof
        try:
            for argv in (["--cwd", str(self.cwd)], ["--cwd", str(self.cwd), "--mode", "advisor"],
                         ["--cwd", str(self.cwd), "--mode=advisor"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    lowctx.main(argv, header=lambda ag: seen.append(ag.mode) or ag.transcript.close())
        finally:
            lowctx.read_input = orig
        self.assertEqual(seen, ["code", "advisor", "advisor"])


if __name__ == "__main__":
    unittest.main()
