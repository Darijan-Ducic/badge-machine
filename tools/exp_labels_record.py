"""Record webcam-emulated sessions through the real Dictation page with NO server behind it: each run gets its own
in-process stub (static/ files, /sam -> no faces, /sessions -> none, POST /dictation -> saved), so nothing reaches
Atlas or a model. Arms: "default" (Chrome's echo cancellation, noise suppression, auto gain on, as shipped) and
"raw" (?audio=raw, all three off).

  uv run python tools/exp_labels_record.py --arm default --out DIR SESSION_DIR/NAME ... [-j 4]

Writes DIR/<name>.<arm>.webm and DIR/<name>.<arm>.run.json (the driver's report: track settings, timing).
"""

import argparse
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "exp:lab-labels-raw:practice:obama-weekly:obama"


def stub(out: Path) -> ThreadingHTTPServer:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(ROOT / "static"), **k)

        def log_message(self, *a):
            pass

        def _json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/sessions"):
                return self._json({"users": [], "sessions": []})
            return super().do_GET()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.startswith("/dictation"):
                out.write_bytes(body)
                return self._json({"session": "stub", "stt": "(stub)", "turns": 0})
            return self._json({"objects": []})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def record(session: Path, arm: str, out_dir: Path) -> dict:
    out = out_dir / f"{session.name}.{arm}.webm"
    server = stub(out)
    page = "/dictation.html" + ("?audio=raw" if arm == "raw" else "")
    try:
        p = subprocess.run(["node", str(ROOT / "tools/emulate_webcam_session.mjs"), "--video", f"{session}.y4m",
                            "--audio", f"{session}.wav", "--out", str(out), "--server",
                            f"http://127.0.0.1:{server.server_address[1]}", "--page", page, "--user", USER,
                            "--control", "space:#hold"],
                           capture_output=True, text=True, timeout=300)
    finally:
        server.shutdown()
    report = json.loads(p.stdout) if p.stdout.strip().startswith("{") else {"error": p.stderr[-800:]}
    (out_dir / f"{session.name}.{arm}.run.json").write_text(json.dumps(report, indent=1))
    ok = out.exists() and out.read_bytes()[:4] == b"\x1a\x45\xdf\xa3"
    return {"session": session.name, "arm": arm, "ok": ok, "audio": report.get("tracks", {}).get("audio"),
            "error": report.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--arm", choices=["default", "raw"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-j", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.j) as pool:
        names = [n for s in args.sessions for n in (Path(s[1:]).read_text().split() if s.startswith("@") else [s])]
        for r in pool.map(lambda s: record(Path(s), args.arm, out), names):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
