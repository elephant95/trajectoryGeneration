"""Phase 0: scene builder, reveal engine, puzzles, and the encoded Reel."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from geoquiz import puzzles
from geoquiz.reel import ReelSpec, render_reel
from geoquiz.render.reveal import RevealStack, build_stack, time_map
from geoquiz.scene import (Frame, LineLayer, View, network_distance, normalise_times, outline_times, stroke_widths,
                           upstream_distance)
from geoquiz.schema import Feature

SMALL = Frame(width=270, height=480, fps=10)


def test_view_fits_extent_inside_box():
    v = View.fit((0, 0, 2000, 1000), (100, 200, 500, 800))
    px = v.px(np.array([[0, 0], [2000, 1000], [1000, 500]]))
    assert px[:, 0].min() == pytest.approx(100) and px[:, 0].max() == pytest.approx(500)
    assert px[2] == pytest.approx([300, 500])                # centred; y grows downward
    assert px[0, 1] > px[1, 1]


def test_network_distance_follows_the_network():
    # an L: (0,0)->(100,0)->(100,100), plus a disconnected stub near the origin
    xy = np.array([[0, 0], [100, 0], [100, 0], [100, 100], [0, 10], [0, 20]], float)
    start = np.array([0, 2, 4, 6])
    d = network_distance(xy, start, origin=(0, 0))
    assert d[:4] == pytest.approx([0, 100, 100, 200])        # around the corner, not the 141 m diagonal
    assert d[4] == pytest.approx(14.0) and d[5] == pytest.approx(28.0)   # unreachable: straight line x detour


def test_upstream_distance_grows_from_basin_exit():
    # two reaches in flow order: A (upstream) then B (downstream, 0 km from the outlet), each 100 m
    xy = np.array([[200, 0], [100, 0], [100, 0], [0, 0]], float)
    d = upstream_distance(xy, np.array([0, 2, 4]), dist_dn_km=np.array([0.1, 0.0]), basin=np.array([1, 1]))
    assert d == pytest.approx([200, 100, 100, 0])
    assert normalise_times(d, 200) == pytest.approx([1, 0.5, 0.5, 0])
    assert stroke_widths(np.array([0.0, 1.0]), 1, 5) == pytest.approx([1, 5])
    assert outline_times(xy, np.array([0, 2, 4])) == pytest.approx([0, 1, 0, 1])


def _line_layer():
    xy = np.array([[20, 240], [250, 240]], np.float32)
    return LineLayer("l", Feature.STREET, xy, np.array([0, 2]), np.array([4.0], np.float32),
                     np.array([0.0, 1.0], np.float32), (1.0, 0.5, 0.0))


def test_time_map_increases_along_the_line():
    T = time_map(SMALL, [_line_layer()])
    row = T[240]
    assert np.isinf(T[100]).all()
    assert row[30] < row[130] < row[240] and row[30] < 0.1 and row[240] > 0.9


def test_reveal_stack_is_empty_then_grows_to_the_full_image():
    stack = build_stack(SMALL, [_line_layer()], fade=0.05, head=0.5)
    assert 0 < stack.ink_fraction < 0.05
    assert not stack.frame(-0.01).any()
    half = stack.frame(0.5)
    assert half[240, 60].any() and not half[240, 220].any()            # drawn up to the halfway point only
    lit = [int(stack.frame(p).any(axis=2).sum()) for p in (0.6, 0.8, stack.end)]
    assert lit == sorted(lit) and lit[-1] == round(stack.ink_fraction * SMALL.width * SMALL.height)
    assert stack.frame(stack.end)[240, 130].tolist() == [255, 128, 0]  # final frame is exactly the crisp render
    assert stack.moving_fraction(0.5) < stack.ink_fraction
    with pytest.raises(ValueError):
        stack.frame(0.2)
    stack.reset()
    assert not stack.frame(0.0)[240, 220].any()


def test_reveal_stack_handles_pixels_missing_from_time_map():
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[1, 1] = (10, 20, 30, 255)
    stack = RevealStack(rgba, np.full((4, 4), np.inf, np.float32), fade=0.1)
    assert not stack.frame(0.5).any() and stack.frame(stack.end)[1, 1].tolist() == [10, 20, 30]


def test_country_puzzle(registry, resolver):
    p = puzzles.country_rivers(registry, resolver, "SQL", frame=SMALL)
    assert p.slug == "SQL_rivers_neon" and p.texts.answer == "Squareland"
    assert [l.name for l in p.scene.layers] == ["rivers", "lakes"] and p.scene.answer_layers[0].name == "border"
    rivers = p.scene.layers[0]
    assert rivers.time.min() == 0 and rivers.time.max() == 1
    trunk = rivers.width > rivers.width.min()
    x = np.array([rivers.xy[s:e, 0].mean() for s, e in zip(rivers.start[:-1], rivers.start[1:])])
    t = np.array([rivers.time[s:e].mean() for s, e in zip(rivers.start[:-1], rivers.start[1:])])
    assert np.corrcoef(x[trunk], t[trunk])[0, 1] > 0.95               # the river drains west: it grows from the west
    assert p.texts.credit == "HydroSHEDS © WWF"                        # only the layers drawn are credited
    assert p.scene.stats["river_km_inside"] > 0 and f"{p.scene.stats['river_km_inside']:,.0f} km" in p.texts.fact
    assert 0.15 < p.scene.stats["frame_fill"] < 0.5
    assert p.gain.max() <= 1 and p.gain[SMALL.height // 2, SMALL.width // 2] == pytest.approx(1, abs=0.01)
    cut = puzzles.country_rivers(registry, resolver, "SQL", frame=SMALL, outside="cut")
    assert cut.gain[int(SMALL.height * 0.49), 2] == 0 < p.gain[int(SMALL.height * 0.49), 14]


def test_city_puzzle(registry, resolver):
    p = puzzles.city_streets(registry, resolver, "Gridville", frame=SMALL, budget_km=60)
    streets = p.scene.layers[-1]
    assert streets.name == "streets" and p.texts.answer == "Gridville" and p.answer["iso_a2"] == "CH"
    centre = np.array([SMALL.width / 2, (530 + 1400) / 2 / 1920 * SMALL.height])
    r = np.hypot(*(streets.xy - centre).T)
    assert streets.time[r < 10].max() < 0.2 < streets.time[r > 100].mean()     # grows outward from the centre
    assert p.texts.credit == "© OpenStreetMap contributors"
    assert [s["source"] for s in p.scene.sources] == ["osm", "ghsl"]
    assert p.scene.stats["street_km_in_window"] == pytest.approx(60, rel=0.2)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed")
def test_reel_end_to_end(registry, resolver, tmp_path):
    p = puzzles.country_rivers(registry, resolver, "SQL", frame=SMALL)
    meta = render_reel(p, tmp_path, land=registry.get("natural_earth").land(), spec=ReelSpec(duration=3.0), index=7)
    folder = tmp_path / "0007_SQL_rivers_neon"
    for name in ("video.mp4", "cover.jpg", "caption.txt", "answer.json", "metadata.json", "data/rivers.parquet"):
        assert (folder / name).stat().st_size > 0, name
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(folder / "video.mp4")],
        capture_output=True, text=True, check=True).stdout)["streams"]
    video = next(s for s in probe if s["codec_type"] == "video")
    assert (video["codec_name"], video["pix_fmt"], video["width"], video["height"]) == ("h264", "yuv420p", 270, 480)
    assert int(video["nb_frames"]) == 30 and any(s["codec_name"] == "aac" for s in probe)
    data = (folder / "video.mp4").read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")                 # faststart
    stored = json.loads((folder / "metadata.json").read_text())
    assert stored["params_hash"] == meta["params_hash"] and len(stored["params_hash"]) == 64
    assert stored["sources"][0]["licence"]["id"] == "LicenseRef-HydroSHEDS-v1"
    assert stored["attribution"]["on_screen"] == "HydroSHEDS © WWF" and 0 < stored["qc"]["ink_fraction"] < 0.35
    assert "World Wildlife Fund" in (folder / "caption.txt").read_text()
    assert json.loads((folder / "answer.json").read_text())["answer"] == "Squareland"
