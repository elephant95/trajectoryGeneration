"""A local point-and-click interface, served to your own browser.

    geoquiz gui            # opens http://127.0.0.1:8765

Pick a continent, a country or city, the features and what to make (a video,
laser-engraving files, or both); the page shows progress and the results.
The server listens on this computer only and uses nothing beyond the
standard library.
"""
from __future__ import annotations

import contextlib
import json
import mimetypes
import os
import subprocess
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import cli
from .regions import fold

PAGE = Path(__file__).with_name("gui.html")


class _Log:
    """File-like object that collects printed lines for the running job."""

    def __init__(self, job: dict):
        self.job, self.buf = job, ""

    def write(self, text: str) -> int:
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.job["log"].append(line.rstrip())
        return len(text)

    def flush(self) -> None:
        pass


class App:
    def __init__(self, data: str | None = None):
        import argparse
        self.ctx = cli._context(argparse.Namespace(data=data))
        self.paths, self.store, self.raw, self.registry, self.resolver = self.ctx
        self.jobs: dict[int, dict] = {}
        self.lock = threading.Lock()
        self.busy = threading.Lock()          # one job at a time: they share the data store and stdout

    # -- lookups ----------------------------------------------------------------------
    def meta(self) -> dict:
        from .puzzles import DEFAULT_FEATURES, LAYERS, OUTSIDE_MODES, REVEAL_MODES
        from .render.style import STYLES
        from .schema import Feature
        return {
            "styles": [{"name": s.name, "paper": s.mode == "subtract",
                        "bg": s.background_centre, "fg": s.color(Feature.RIVER)} for s in STYLES.values()],
            "reveals": list(REVEAL_MODES), "outside": list(OUTSIDE_MODES),
            "features": {k.value: [{"name": s.name, "label": s.label} for s in v.values()] for k, v in LAYERS.items()},
            "default_features": {k.value: list(v) for k, v in DEFAULT_FEATURES.items()},
            "defaults": cli.RENDER_DEFAULTS, "out": str(self.paths.out),
            "have_cities": self.store.has("ghsl", Feature.URBAN_CENTRE),
        }

    def countries(self) -> list[dict]:
        t = self.registry.get("natural_earth").countries()
        t = t[t["x_pop"] > 0].sort_values("name")
        return [{"iso3": r["source_id"], "name": r["name"], "continent": r["x_continent"], "population": float(r["x_pop"])}
                for _, r in t.iterrows()]

    def city_countries(self) -> list[dict]:
        """Countries that have cities, with the continent each belongs to (for the city picker)."""
        cities = self.registry.get("ghsl").urban_centres()
        continent = {fold(c["name"]): c["continent"] for c in self.countries()}
        counts = cities.groupby("x_country").size()
        return [{"name": name, "continent": continent.get(fold(name), "Other"), "cities": int(n)}
                for name, n in sorted(counts.items())]

    def cities(self, country: str) -> list[dict]:
        t = self.registry.get("ghsl").urban_centres()
        t = t[t["x_country"].map(fold) == fold(country)].sort_values("x_pop", ascending=False)
        t = t[t["name"].notna()]                          # a few urban centres have no name: they cannot be asked for
        return [{"name": str(r["name"]), "population": float(r["x_pop"])} for _, r in t.iterrows()]

    def features(self, kind: str, target: str, country: str | None) -> list[dict]:
        st = cli.feature_status(self.ctx, kind, target, country)["features"]
        return [{"name": k, "label": v["label"], "ready": bool(v["ready"]), "source": v["source"]} for k, v in st.items()]

    # -- jobs ---------------------------------------------------------------------------
    def start(self, fn, title: str) -> dict:
        with self.lock:
            if self.busy.locked():
                return {"error": "Something is still running. Wait for it to finish."}
            jid = len(self.jobs) + 1
            job = self.jobs[jid] = {"id": jid, "title": title, "state": "running", "log": [], "result": None, "error": None}
            self.busy.acquire()

        def work():
            try:
                with contextlib.redirect_stdout(_Log(job)):
                    job["result"] = fn()
                job["state"] = "done"
            except Exception as e:                       # shown to the user in the page
                job["error"] = str(e.args[0]) if e.args else repr(e)
                job["log"].append(traceback.format_exc(limit=3).strip().splitlines()[-1])
                job["state"] = "failed"
            finally:
                self.busy.release()

        threading.Thread(target=work, daemon=True).start()
        return {"id": jid}

    def download_cities(self) -> dict:
        from .auto import ensure_basics
        return self.start(lambda: {"downloaded": ensure_basics(self.registry, self.raw, cities=True)}, "Downloading the city list")

    def make(self, req: dict) -> dict:
        kind, target, outputs = req["kind"], req["target"], set(req.get("outputs", ["video"]))
        if not outputs & {"video", "engrave"}:
            return {"error": "Choose at least one thing to make: a video or engraving files."}
        settings = {k: v for k, v in req.get("settings", {}).items() if v not in ("", None)}
        unknown = set(settings) - set(cli.RENDER_DEFAULTS)
        if unknown:
            return {"error": f"unknown settings {sorted(unknown)}"}
        settings["ingest_missing"] = bool(req.get("ingest_missing", True))

        def run():
            out = self.paths.out
            result = {"target": target, "files": {}}
            if "video" in outputs:
                meta = cli.render_one(self.ctx, kind, target, {**settings, "engrave": "engrave" in outputs}, out)
                folder = Path(meta["folder"])
                result.update(folder=str(folder), video=self._url(folder / "video.mp4"), cover=self._url(folder / "cover.jpg"),
                              caption=(folder / "caption.txt").read_text(encoding="utf-8"), credit=meta["attribution"]["on_screen"])
                engrave = meta.get("engrave")
            else:
                engrave = cli.engrave_one(self.ctx, kind, target, settings, out)
                result["folder"] = engrave["folder"]
            if engrave:
                ef = Path(engrave["folder"])
                result["engrave"] = {
                    "folder": str(ef), "canvas_mm": engrave["canvas_mm"], "scale": engrave["scale"],
                    "layers": [{"name": n, "svg": self._url(ef / f"{n}.svg"), "png": self._url(ef / f"{n}.png")}
                               for n in ["all", *engrave["layers"], "contour"]]}
            return result

        return self.start(run, f"{kind} {target}")

    def _url(self, path: Path) -> str:
        return "/files/" + path.resolve().relative_to(self.paths.out.resolve()).as_posix()

    def open_folder(self, folder: str) -> dict:
        p = Path(folder).resolve()
        if not p.is_dir() or not p.is_relative_to(self.paths.out.resolve()):
            return {"error": "not an output folder"}
        if sys.platform == "win32":
            os.startfile(p)                                              # noqa: S606 - a folder we created
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])
        return {"ok": True}


def _clean(obj):
    """NaN is not valid JSON: send null instead."""
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):                    # keep the terminal quiet
            pass

        def _json(self, obj, status: int = 200):
            body = json.dumps(_clean(obj), ensure_ascii=False, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path):
            root = app.paths.out.resolve()
            path = path.resolve()
            if not path.is_file() or not path.is_relative_to(root):
                return self._json({"error": "not found"}, 404)
            size = path.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):          # video players seek with range requests
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(size - int(b), 0)
                end = min(int(b), size - 1) if a and b else size - 1
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    body = PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/api/meta":
                    self._json(app.meta())
                elif url.path == "/api/countries":
                    self._json(app.countries())
                elif url.path == "/api/city-countries":
                    self._json(app.city_countries())
                elif url.path == "/api/cities":
                    self._json(app.cities(q["country"]))
                elif url.path == "/api/features":
                    self._json(app.features(q["kind"], q["target"], q.get("country")))
                elif url.path.startswith("/api/job/"):
                    job = app.jobs.get(int(url.path.rsplit("/", 1)[1]))
                    self._json(job or {"error": "no such job"}, 200 if job else 404)
                elif url.path.startswith("/files/"):
                    self._file(app.paths.out / unquote(url.path[len("/files/"):]))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": str(e.args[0]) if e.args else repr(e)}, 400)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/make":
                    self._json(app.make(req))
                elif self.path == "/api/download-cities":
                    self._json(app.download_cities())
                elif self.path == "/api/open-folder":
                    self._json(app.open_folder(req.get("folder", "")))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": str(e.args[0]) if e.args else repr(e)}, 400)

    return Handler


def serve(data: str | None = None, port: int = 8765, open_browser: bool = True) -> None:
    from .auto import ensure_basics
    app = App(data)
    ensure_basics(app.registry, app.raw)                 # the country list, on first use
    server = None
    for p in range(port, port + 20):                     # the next free port if the usual one is taken
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), _handler(app))
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port between {port} and {port + 19}")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"geoquiz is running at {url}  (close this window or press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
