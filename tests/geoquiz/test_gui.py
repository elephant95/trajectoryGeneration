"""The local browser interface: its server, driven over HTTP against the synthetic store."""
import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from geoquiz.gui import PAGE, App, _handler


@pytest.fixture()
def server(store):
    app = App(str(store.root.parent))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", app
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.headers, r.read()


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _wait(base, jid, seconds=60):
    for _ in range(seconds * 10):
        job = json.loads(_get(f"{base}/api/job/{jid}")[2])
        if job["state"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_page_and_lookups(server):
    base, _ = server
    status, headers, body = _get(base + "/")
    assert status == 200 and b"<title>geoquiz</title>" in body and body == PAGE.read_bytes()
    meta = json.loads(_get(base + "/api/meta")[2])
    assert [s["name"] for s in meta["styles"]][:2] == ["neon", "ink"] and meta["have_cities"]
    assert {f["name"] for f in meta["features"]["city"]} == {"water", "waterways", "buildings", "streets", "rail"}
    countries = json.loads(_get(base + "/api/countries")[2])
    assert [c["iso3"] for c in countries] == ["OTH", "SQL"] and countries[1]["continent"] == "Europe"
    assert json.loads(_get(base + "/api/city-countries")[2]) == [{"name": "Squareland", "continent": "Europe", "cities": 1}]
    assert json.loads(_get(base + "/api/cities?country=squareland")[2]) == [{"name": "Gridville", "population": 120000.0}]
    feats = {f["name"]: f for f in json.loads(_get(base + "/api/features?kind=country&target=SQL")[2])}
    assert feats["rivers"]["ready"] and feats["rivers"]["source"] == "hydrosheds" and not feats["rail"]["ready"]
    with pytest.raises(urllib.error.HTTPError) as e:                       # a bad request is a JSON error, not a crash
        _get(base + "/api/features?kind=country&target=Atlantis")
    assert "no country matches" in json.loads(e.value.read())["error"]


def test_make_engraving_and_serve_files(server):
    base, app = server
    res = _post(base + "/api/make", {"kind": "country", "target": "SQL", "outputs": ["engrave"], "ingest_missing": False,
                                     "settings": {"features": ["rivers", "lakes"], "width_mm": 80, "dpi": 127, "hint": ""}})
    job = _wait(base, res["id"])
    assert job["state"] == "done", job
    e = job["result"]["engrave"]
    assert e["canvas_mm"][0] == 80 and [l["name"] for l in e["layers"]] == ["all", "rivers", "lakes", "contour"]
    assert any("80 x" in line for line in job["log"])                       # progress lines reach the page
    status, headers, body = _get(base + e["layers"][1]["svg"])
    assert status == 200 and headers["Content-Type"].startswith("image/svg") and b'id="rivers"' in body
    req = urllib.request.Request(base + e["layers"][0]["png"], headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req) as r:
        assert r.status == 206 and r.read() == b"\x89PNG\r\n\x1a\n" and r.headers["Content-Range"].startswith("bytes 0-7/")
    # only files inside the output folder are served
    for path in ("/files/../store/_licences.json", "/files/%2e%2e/%2e%2e/etc/passwd"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + path)
        assert err.value.code == 404
    assert "error" in app.open_folder(str(app.paths.root))


def test_make_reports_problems(server):
    base, _ = server
    assert "Choose at least one" in _post(base + "/api/make", {"kind": "country", "target": "SQL", "outputs": []})["error"]
    assert "unknown settings" in _post(base + "/api/make", {"kind": "country", "target": "SQL", "settings": {"colour": 1}})["error"]
    res = _post(base + "/api/make", {"kind": "country", "target": "SQL", "outputs": ["engrave"], "ingest_missing": False,
                                     "settings": {"features": ["rail"]}})
    job = _wait(base, res["id"])
    assert job["state"] == "failed" and "no data for rail" in job["error"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_make_video_with_engraving(server):
    base, _ = server
    res = _post(base + "/api/make", {"kind": "city", "target": "Gridville", "outputs": ["video", "engrave"], "ingest_missing": False,
                                     "settings": {"features": ["streets"], "country": "Squareland", "budget_km": 60, "style": "ink",
                                                  "size": "180x320", "fps": 10, "duration": 2, "width_mm": 60, "dpi": 127}})
    job = _wait(base, res["id"])
    assert job["state"] == "done", job
    r = job["result"]
    assert r["video"].endswith("Gridville_streets_ink/video.mp4") and r["credit"] == "© OpenStreetMap contributors"
    status, headers, body = _get(base + r["video"])
    assert headers["Content-Type"] == "video/mp4" and body[4:8] == b"ftyp"
    assert r["engrave"]["layers"][1]["svg"].endswith("/engrave/streets.svg")


def test_make_all_reports_each_place_and_can_stop(server):
    base, app = server
    targets = [{"target": "SQL", "name": "Squareland"}, {"target": "OTH", "name": "Otherland"}]
    body = {"kind": "country", "targets": targets, "where": "Europe", "outputs": ["engrave"], "ingest_missing": False,
            "settings": {"features": ["rivers", "lakes", "coast"], "width_mm": 60, "dpi": 127, "answer": "ignored"}}
    job = _wait(base, _post(base + "/api/make-all", body)["id"])
    assert job["state"] == "done" and job["progress"] == {"done": 2, "total": 2, "current": None}
    r = job["result"]
    assert r["batch"] and [i["name"] for i in r["items"]] == ["Squareland"] and not r["stopped"]
    assert [f["name"] for f in r["failed"]] == ["Otherland"] and "no data" in r["failed"][0]["error"]   # reported, not fatal
    item = r["items"][0]
    assert item["folder"].endswith("all_europe_country/0001_SQL_rivers-lakes_engrave")
    assert _get(base + item["engrave"]["layers"][0]["svg"])[0] == 200
    assert "Nothing matches" in _post(base + "/api/make-all", {**body, "targets": []})["error"]
    assert "nothing to stop" in _post(base + "/api/cancel", {"id": job["id"]})["error"]
    # a stop request ends the run before the next place
    res = app.make_all({**body, "targets": targets * 3})
    app.jobs[res["id"]]["cancel"] = True
    job = _wait(base, res["id"])
    assert job["result"]["stopped"] and len(job["result"]["items"]) + len(job["result"]["failed"]) < 6
