#!/usr/bin/env python3
"""Local web server for the GTO trainer.  Run:  python3 server.py  then open http://localhost:8765"""
import argparse
import json
import mimetypes
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from gto import config, netflop
from gto.engine import Trainer

trainer = Trainer()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the terminal quiet; errors still print via traceback
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            return self._json(trainer.snapshot())
        if path == "/api/stats":
            return self._json(trainer.history.summary())
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.normpath(os.path.join(config.STATIC_DIR, rel))
        if not full.startswith(config.STATIC_DIR) or not os.path.isfile(full):
            return self._json({"error": "not found"}, 404)
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(full)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/new_hand":
                trainer.new_hand(body.get("matchup", "random"), body.get("hero", "random"), body.get("flop_mode", "cache"))
            elif path == "/api/act":
                trainer.act(int(body["index"]))
            elif path == "/api/reset_stats":
                trainer.reset_stats()
            else:
                return self._json({"error": "not found"}, 404)
            return self._json({"ok": True})
        except (RuntimeError, ValueError, KeyError) as exc:
            return self._json({"error": str(exc)}, 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(config.SOLVER_BIN):
        sys.exit(f"solveur introuvable: {config.SOLVER_BIN}\n  -> cd tools/turn-labels && cargo build --release")
    try:
        netflop.preload()                    # the "main IRL" mode needs torch: fine to be without it, the other modes still work
    except netflop.Unavailable as exc:
        print(f"Mode « Main IRL » désactivé : {exc}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"GTO Trainer sur {url}  (Ctrl-C pour arrêter)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
