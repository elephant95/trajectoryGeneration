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


def _path_means(layer):
    x = np.array([layer.xy[s:e, 0].mean() for s, e in zip(layer.start[:-1], layer.start[1:])])
    t = np.array([layer.time[s:e].mean() for s, e in zip(layer.start[:-1], layer.start[1:])])
    return x, t


def test_country_puzzle_hides_the_shape_by_default(registry, resolver):
    p = puzzles.build_puzzle(registry, resolver, "country", "SQL", frame=SMALL)
    assert p.slug == "SQL_rivers-lakes_neon" and p.feature == "rivers+lakes" and p.texts.answer == "Squareland"
    assert [l.name for l in p.scene.layers] == ["rivers", "lakes"] and p.scene.answer_layers[0].name == "border"
    rivers = p.scene.layers[0]
    assert rivers.time.min() == 0 and rivers.time.max() == 1
    x, t = _path_means(rivers)
    trunk = rivers.width > np.median(rivers.width)
    assert np.corrcoef(x[trunk], t[trunk])[0, 1] > 0.95               # the river drains west: it grows from the west
    assert p.texts.credit == "HydroSHEDS © WWF"                        # only the layers drawn are credited
    assert p.scene.stats["rivers_km"] > 0 and f"{p.scene.stats['rivers_km']:,.0f} km" in p.texts.fact
    # outside="keep": the country sits inside a full rectangle of features at full brightness
    border_x = p.scene.answer_layers[0].xy[:, 0]
    assert rivers.xy[:, 0].min() < border_x.min() - 10 and rivers.xy[:, 0].max() > border_x.max() + 10
    mid_y = int(p.scene.answer_layers[0].xy[:, 1].mean())
    assert p.gain[mid_y, int(border_x.min()) - 8] == pytest.approx(p.gain[mid_y, SMALL.width // 2], abs=0.05)
    assert p.params["outside"] == "keep"


def test_outside_dim_and_cut_reveal_the_shape(registry, resolver):
    dim = puzzles.build_puzzle(registry, resolver, "country", "SQL", frame=SMALL, options=puzzles.PuzzleOptions(outside="dim"))
    cut = puzzles.build_puzzle(registry, resolver, "country", "SQL", frame=SMALL, options=puzzles.PuzzleOptions(outside="cut"))
    y = int(dim.scene.answer_layers[0].xy[:, 1].mean())
    assert dim.gain[y, SMALL.width // 2] == pytest.approx(1, abs=0.01)
    assert cut.gain[y, 2] == 0 < dim.gain[y, 14] < 0.5
    assert 0.15 < dim.scene.stats["frame_fill"] < 0.5
    with pytest.raises(ValueError):
        puzzles.PuzzleOptions(outside="hide")


def test_any_feature_mix_and_reveal_mode(registry, resolver):
    p = puzzles.build_puzzle(registry, resolver, "country", "SQL", ("roads", "cities", "coast"), SMALL,
                             options=puzzles.PuzzleOptions(reveal="sweep"))
    assert [l.name for l in p.scene.layers] == ["coast", "roads", "cities"]          # drawn back to front
    assert p.feature == "roads+cities+coast" and p.texts.hint == "Guess it from its main roads, cities and coastline"
    roads = next(l for l in p.scene.layers if l.name == "roads")
    assert np.corrcoef(roads.xy[:, 0], roads.time)[0, 1] > 0.99                       # west to east
    assert p.accent == p.style.color(Feature.STREET)
    assert "km of main roads inside its borders" in p.texts.fact
    assert p.texts.credit == "© OpenStreetMap contributors · GHSL © European Union · Natural Earth"
    cities = puzzles.build_puzzle(registry, resolver, "country", "SQL", ("cities",), SMALL)
    assert cities.texts.fact.startswith("1 cities of 50,000") and cities.scene.stats["cities_count"] == 1
    ranked = puzzles.build_puzzle(registry, resolver, "country", "SQL", ("rivers",), SMALL,
                                  options=puzzles.PuzzleOptions(reveal="rank"))
    r = ranked.scene.layers[0]
    _, t = _path_means(r)
    wide = r.width > np.median(r.width)
    assert t[wide].max() < t[~wide].min()                                              # big rivers first


def test_missing_data_and_bad_input_are_explained(registry, resolver):
    with pytest.raises(puzzles.MissingData, match="geoquiz ingest --sources overture --country"):
        puzzles.build_puzzle(registry, resolver, "country", "SQL", ("rail",), SMALL)
    with pytest.raises(ValueError, match="choose from"):
        puzzles.build_puzzle(registry, resolver, "country", "SQL", ("streets",), SMALL)
    with pytest.raises(puzzles.MissingData):
        puzzles.build_puzzle(registry, resolver, "city", "Gridville", ("buildings",), SMALL)
    # defaults skip what is absent instead of failing (no rail, water or waterways in this store)
    p = puzzles.build_puzzle(registry, resolver, "city", "Gridville", frame=SMALL)
    assert p.feature == "streets"


def test_text_overrides(registry, resolver):
    o = puzzles.PuzzleOptions(question=("Wo ist das?",), hint="Tipp", answer="Quadratland", fact="Eine Tatsache")
    p = puzzles.build_puzzle(registry, resolver, "country", "SQL", frame=SMALL, options=o)
    assert (p.texts.question, p.texts.hint, p.texts.answer, p.texts.fact) == (("Wo ist das?",), "Tipp", "Quadratland", "Eine Tatsache")
    assert p.answer["source_name"] == "Squareland" and p.answer["fact_source"] == "supplied"


def test_thinning_is_even_across_the_frame():
    rng = np.random.default_rng(0)
    n = 4000
    mid = np.column_stack([rng.uniform(0, 1200, n), rng.uniform(0, 1000, n)])
    rank = np.where(mid[:, 0] < 600, rng.uniform(0.5, 1, n), rng.uniform(0, 0.5, n))   # the west outranks the east
    keep = puzzles._thin_by_budget(rank, np.arange(n), np.full(n, 10.0), mid, (0, 0, 1200, 1000), budget_px=12_000)
    west, east = keep[mid[:, 0] < 600].sum(), keep[mid[:, 0] >= 600].sum()
    assert 0.8 < west / east < 1.25 and 1100 < keep.sum() < 1300
    assert rank[keep & (mid[:, 0] >= 600)].mean() > rank[~keep & (mid[:, 0] >= 600)].mean()
    # a network keeps whole classes: the top class always, the next while it fits
    cls = np.array([1.0, 1.0, 0.8, 0.8, 0.6])
    assert puzzles._thin_by_level(cls, np.arange(5), np.full(5, 10.0), 45).tolist() == [True] * 4 + [False]
    assert puzzles._thin_by_level(cls, np.arange(5), np.full(5, 10.0), 5).tolist() == [True, True, False, False, False]


def test_city_puzzle(registry, resolver):
    p = puzzles.build_puzzle(registry, resolver, "city", "Gridville", ("streets",), SMALL,
                             options=puzzles.PuzzleOptions(budget_km=60))
    streets = p.scene.layers[-1]
    assert streets.name == "streets" and p.texts.answer == "Gridville" and p.answer["iso_a2"] == "CH"
    centre = np.array([SMALL.width / 2, (530 + 1400) / 2 / 1920 * SMALL.height])
    r = np.hypot(*(streets.xy - centre).T)
    assert streets.time[r < 10].max() < 0.2 < streets.time[r > 100].mean()     # grows outward from the centre
    assert p.texts.credit == "© OpenStreetMap contributors"
    assert [s["source"] for s in p.scene.sources] == ["osm", "ghsl"]
    assert p.scene.stats["streets_km"] == pytest.approx(60, rel=0.2)
    fixed = puzzles.build_puzzle(registry, resolver, "city", "Gridville", ("streets",), SMALL,
                                 options=puzzles.PuzzleOptions(radius_km=0.5))
    assert fixed.scene.stats["window_km"][0] == pytest.approx(1.0, abs=0.01)


def test_subtractive_style_darkens_a_light_background():
    from geoquiz.render.reveal import compose
    from geoquiz.render.style import INK, STYLES, with_overrides
    stack = build_stack(SMALL, [_line_layer()], fade=0.05, head=0.0, invert=True)
    bg = np.full((SMALL.height, SMALL.width, 3), 240, np.uint8)
    out = compose(stack.frame(stack.end), bg, 0.0, 0.0, mode="subtract")
    assert out[240, 130, 0] == 240 and out[240, 130, 1] in (112, 113) and out[240, 130, 2] == 0 and out[100, 100].tolist() == [240, 240, 240]   # orange ink on paper
    assert INK.mode == "subtract" and set(STYLES) == {"neon", "ink", "blueprint", "ember", "mono"}
    assert with_overrides(INK, {"river": [1, 0, 0]}).color(Feature.RIVER) == (1, 0, 0)


def test_timeline_from_phases():
    from geoquiz.render.overlay import Timeline
    tl = Timeline.from_phases(10, 0, 4, 1)
    assert (tl.duration, tl.reveal_end, tl.countdown_end, tl.answer_end) == (15, 10, 10, 14)
    with pytest.raises(ValueError):
        Timeline.from_phases(0, 3, 3, 1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed")
def test_reel_end_to_end(registry, resolver, tmp_path):
    p = puzzles.country_rivers(registry, resolver, "SQL", frame=SMALL)
    meta = render_reel(p, tmp_path, land=registry.get("natural_earth").land(), spec=ReelSpec(duration=3.0), index=7)
    folder = tmp_path / "0007_SQL_rivers-lakes_neon"
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_batch_config(store, tmp_path, capsys):
    from geoquiz.cli import main
    cfg = tmp_path / "batch.yaml"
    cfg.write_text("""
defaults: {size: 180x320, fps: 10, phases: "1,0,1,0.5", style: ink}
puzzles:
  - {kind: country, target: SQL, features: [rivers, lakes], question: "Which land|is this?"}
  - {kind: city, target: Gridville, features: streets, style: mono, budget_km: 60}
  - {kind: country, target: SQL, features: [rail]}
random:
  - {kind: country, count: 2, seed: 3, in: Europe, features: cities}
""")
    code = main(["--data", str(store.root.parent), "batch", str(cfg), "--out", str(tmp_path / "out")])
    made = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert made[:2] == ["0001_SQL_rivers-lakes_ink", "0002_Gridville_streets_mono"]
    assert len(made) == 3 and made[2].endswith("_SQL_cities_ink")                  # random pick; Otherland has no cities
    printed = capsys.readouterr().out
    assert code == 1 and "SKIPPED country SQL: no data for rail" in printed and "SKIPPED country OTH" in printed
    meta = json.loads((tmp_path / "out" / made[0] / "metadata.json").read_text())
    assert meta["video"]["frames"] == 25 and meta["style"] == "ink" and meta["params"]["outside"] == "keep"
