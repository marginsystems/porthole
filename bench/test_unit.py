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


# Stands in for ssh: '[-o opt]... -- host command', run by /bin/sh in $FAKE_HOME like a remote login shell would.
FAKE_SSH = """#!/bin/sh
while [ "$1" != "--" ]; do shift; done
shift
host=$1; shift
[ "$host" = down ] && { echo "ssh: connect to host down port 22: Connection refused" >&2; exit 255; }
cd "$FAKE_HOME" && exec /bin/sh -c "$*"
"""


class Remote(Base):
    def setUp(self):
        super().setUp()
        self.home = self.cwd / "remote-home"
        self.proj = self.home / "proj"
        (self.proj / "src").mkdir(parents=True)
        (self.proj / "src" / "app.py").write_text("x = 1\nretry_limit = 3\n")
        (self.proj / ".env").write_text("TOKEN=hunter2\n")
        fake = self.cwd / "fake-ssh"
        fake.write_text(FAKE_SSH)
        fake.chmod(0o755)
        self.old = lowctx.SSH, os.environ.get("FAKE_HOME")
        lowctx.SSH, os.environ["FAKE_HOME"] = str(fake), str(self.home)
        self.local = self.cwd / "here"
        self.local.mkdir()
        self.r = lowctx.Agent(self.local, 12000, 20, False, "code", quiet=True, ssh="box:~/proj")
        self.t = self.r.tools

    def tearDown(self):
        self.r.transcript.close()
        lowctx.SSH = self.old[0]
        if self.old[1] is None:
            os.environ.pop("FAKE_HOME", None)
        else:
            os.environ["FAKE_HOME"] = self.old[1]
        super().tearDown()

    def test_target_and_prompt(self):
        self.assertEqual(self.t.root, str(self.proj.resolve()))
        self.assertEqual(self.t.where, f"box:{self.proj.resolve()}")
        self.assertIn("remote machine box", self.r.system()["content"])
        self.assertEqual(lowctx.RemoteTools(self.local, self.r.store, "box", "proj").root, self.t.root)
        self.assertEqual(lowctx.RemoteTools(self.local, self.r.store, "box", str(self.proj)).root, self.t.root)
        self.assertEqual(lowctx.parse_ssh_target("[2001:db8::1]:/srv/app"), ("2001:db8::1", "/srv/app"))
        self.assertEqual(lowctx.parse_ssh_target("me@[fe80::1%en0]"), ("me@fe80::1%en0", ""))
        self.assertEqual(lowctx.parse_ssh_target("me@box:~/a:b"), ("me@box", "~/a:b"))
        for bad in ("", ":x", "-oProxyCommand=x:y", "a b:c"):
            self.assertRaises(ValueError, lowctx.parse_ssh_target, bad)
        self.assertRaises(lowctx.SSHError, lowctx.RemoteTools, self.local, self.r.store, "down", "")
        self.assertRaises(lowctx.SSHError, lowctx.RemoteTools, self.local, self.r.store, "box", "nope")

    def test_file_tools_act_on_the_remote(self):
        t = self.t
        self.assertEqual(t.run("ls", {}), ".env  (14b)\nsrc/")
        self.assertIn("app.py  (22b) is a file", t.run("ls", {"path": "src/app.py"}))
        self.assertIn("does not exist", t.run("ls", {"path": "zzz"}))
        self.assertIn("   2| retry_limit = 3", t.run("read", {"path": "src/app.py"}))
        self.assertIn("is a directory", t.run("read", {"path": "src"}))
        self.assertIn("does not exist", t.run("read", {"path": "nope.txt"}))
        self.assertEqual(t.run("grep", {"pattern": "retry_limit"}), "src/app.py:2:retry_limit = 3")
        self.assertEqual(t.run("grep", {"pattern": "zzz"}), "no matches")
        odd = "it's $HOME `x` \"q\"\n"
        self.assertIn("ok: edited", t.run("edit", {"path": "src/app.py", "old": "x = 1\n", "new": odd}))
        self.assertEqual((self.proj / "src/app.py").read_text(), odd + "retry_limit = 3\n")
        self.assertIn("occurs 0", t.run("edit", {"path": "src/app.py", "old": "zzz", "new": "y"}).replace("not found", "occurs 0"))
        self.assertIn("ok: wrote", t.run("write", {"path": "new dir/it's.txt", "content": "hi\n"}))
        self.assertEqual((self.proj / "new dir" / "it's.txt").read_text(), "hi\n")
        out = t.run("bash", {"cmd": "pwd; echo oops >&2; exit 3"})
        self.assertTrue(out.startswith(f"exit=3\n{self.proj.resolve()}"), out)
        self.assertIn("[stderr]\noops", out)
        self.assertFalse((self.local / "src").exists())  # nothing landed in the local folder
        (self.proj / "locked").mkdir(mode=0o300)  # enter, but not list
        self.assertIn("permission denied", t.run("ls", {"path": "locked"}))
        (self.proj / "locked").chmod(0o700)
        (self.proj / "page.html").write_text("<html><body><p>hello there</p></body></html>")
        os.symlink(self.proj / "page.html", self.proj / "page.txt")  # the type is the resolved file's
        self.assertNotIn("<p>", t.run("read", {"path": "page.txt"}))

    def test_lowctx_paths_stay_local(self):
        t = self.t
        self.assertIn("ok: wrote", t.run("write", {"path": ".lowctx/note.txt", "content": "plum\n"}))
        self.assertEqual((self.local / ".lowctx" / "note.txt").read_text(), "plum\n")
        self.assertFalse((self.proj / ".lowctx").exists())
        self.assertIn("1| plum", t.run("read", {"path": ".lowctx/note.txt"}))
        rel = self.r.save_input("a long paste")
        self.assertIsNone(lowctx.worker_blocked(t, "read", {"path": rel}))
        (self.local / "README.md").write_text("local only\n")
        os.symlink("../README.md", self.local / ".lowctx" / "alias")  # out of .lowctx into the local folder
        self.assertIn("outside", lowctx.worker_blocked(t, "read", {"path": ".lowctx/alias"}))

    def test_workers_stay_inside_and_off_secrets(self):
        t = self.t
        self.assertIsNone(lowctx.worker_blocked(t, "read", {"path": "src/app.py"}))
        self.assertIn("off limits", lowctx.worker_blocked(t, "read", {"path": ".env"}))
        self.assertIn("outside", lowctx.worker_blocked(t, "read", {"path": "/etc/hosts"}))
        self.assertIn("outside", lowctx.worker_blocked(t, "ls", {"path": "../"}))
        (self.home / "id_ed25519").write_text("key")
        os.symlink(self.home, self.proj / "up")  # a link inside the folder pointing out of it
        self.assertIn("outside", lowctx.worker_blocked(t, "read", {"path": "up/id_ed25519"}))
        self.assertNotIn(".env", lowctx.worker_filter(t.run("grep", {"pattern": "TOKEN|retry"})))
        # the file is swapped for a link to a key after the check passed: the read itself refuses it
        (self.proj / "notes.txt").write_text("fine\n")
        self.assertIsNone(lowctx.worker_blocked(t, "read", {"path": "notes.txt"}))
        (self.proj / "notes.txt").unlink()
        os.symlink(self.home / "id_ed25519", self.proj / "notes.txt")
        self.assertIn("outside", t.read_full("notes.txt"))
        (self.proj / "notes.txt").unlink()
        os.symlink(self.proj / ".env", self.proj / "notes.txt")
        self.assertIn("off limits", t.read_full("notes.txt"))
        (self.proj / "notes.txt").unlink()
        os.symlink(self.home / "id_ed25519", self.proj / "notes.txt")  # grep names the real file, and the filter drops it
        out = t.run("grep", {"pattern": "key", "path": "notes.txt"})
        self.assertIn("id_ed25519:1:key", out)
        self.assertEqual(lowctx.worker_filter(out), "no matches")
        self.assertEqual(t.run("grep", {"pattern": "retry", "path": "src/app.py"}), "src/app.py:2:retry_limit = 3")
        self.assertIn("retry_limit", t.read_full("src/app.py"))
        self.assertIn("key", t.run("read", {"path": "up/id_ed25519"}))  # the lead itself may still read anything

    def test_remote_session_kept_apart(self):
        self.assertNotEqual(self.r.session_file, self.r.store / "session.json")
        self.r.notes = "remote notes"
        self.r.save_session()
        self.assertFalse((self.r.store / "session.json").exists())


if __name__ == "__main__":
    unittest.main()
