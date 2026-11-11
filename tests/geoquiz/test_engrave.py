"""Laser-engraving export and the Python interface."""
import json
import re

import cv2
import numpy as np
import pytest
import shapely

from geoquiz import GeoQuiz, puzzles
from geoquiz.engrave import export_engraving
from geoquiz.scene import Frame

SMALL = Frame(width=270, height=480, fps=10)


def _coords(svg: str) -> np.ndarray:
    return np.array(re.findall(r"(-?\d+\.\d+),(-?\d+\.\d+)", svg), float)


def test_one_file_per_feature_cut_to_the_contour(registry, resolver, tmp_path):
    tmp_path = tmp_path / "engrave"
    p = puzzles.build_puzzle(registry, resolver, "country", "SQL", ("rivers", "lakes", "roads"), SMALL,
                             options=puzzles.PuzzleOptions(outside="keep"))         # features extend past the border
    m = export_engraving(p, tmp_path, width_mm=120, margin_mm=10, min_line_mm=0.4, dpi=254)
    assert sorted(f.name for f in tmp_path.iterdir()) == sorted(
        ["engrave.json"] + [f"{n}.{e}" for n in ("contour", "rivers", "lakes", "roads", "all") for e in ("svg", "png")])
    assert m["canvas_mm"][0] == 120 and list(m["layers"]) == ["rivers", "lakes", "roads"]
    assert m["layers"]["rivers"]["line_mm"][0] >= 0.4
    w, h = m["canvas_mm"]
    shapes = {}
    for name in ("contour", "rivers", "lakes", "roads", "all"):
        svg = (tmp_path / f"{name}.svg").read_text()
        assert f'width="{w:.3f}mm"' in svg and f'viewBox="0 0 {w:.3f} {h:.3f}"' in svg     # one shared canvas
        c = _coords(svg)
        # nothing outside the contour: every coordinate stays within the margin box
        assert c[:, 0].min() >= 10 - 0.3 and c[:, 0].max() <= w - 10 + 0.3, name
        assert c[:, 1].min() >= 10 - 0.3 and c[:, 1].max() <= h - 10 + 0.3, name
        png = cv2.imread(str(tmp_path / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        assert png.shape == (round(h * 10), 1200) and set(np.unique(png)) == {0, 255}
        shapes[name] = png == 0
    # the layers are separate pictures that add up to the combined one
    assert 'id="rivers"' in (tmp_path / "rivers.svg").read_text() and 'id="lakes"' not in (tmp_path / "rivers.svg").read_text()
    assert shapes["rivers"].sum() > 0 and shapes["lakes"].sum() > 0 and shapes["roads"].sum() > 0
    union = shapes["contour"] | shapes["rivers"] | shapes["lakes"] | shapes["roads"]
    assert (union != shapes["all"]).mean() < 0.002
    # the river reaches the border (it runs on beyond it in the video) but is cut there
    cols = np.flatnonzero(shapes["rivers"].any(axis=0))
    assert 97 <= cols.min() <= 106 and 1094 <= cols.max() <= 1103
    all_svg = (tmp_path / "all.svg").read_text()
    assert all(f'id="{g}"' in all_svg for g in ("rivers", "lakes", "roads", "contour"))
    assert len(set(re.findall(r'(?:fill|stroke)="(#[0-9a-f]{6})"', all_svg))) == 4            # a colour per layer
    assert json.loads((tmp_path / "engrave.json").read_text())["all_svg_colors"]["contour"] == "#000000"


def test_centerline_mirror_and_contour_options(registry, resolver, tmp_path):
    p = puzzles.build_puzzle(registry, resolver, "country", "SQL", ("rivers",), SMALL)
    export_engraving(p, tmp_path / "a", width_mm=100, lines="centerline", contour_in_layers=True, dpi=127)
    svg = (tmp_path / "a" / "rivers.svg").read_text()
    assert 'fill="none"' in svg and "stroke-width=" in svg and 'id="contour"' in svg
    export_engraving(p, tmp_path / "b", width_mm=100, dpi=127)
    export_engraving(p, tmp_path / "c", width_mm=100, dpi=127, mirror=True)
    b = cv2.imread(str(tmp_path / "b" / "rivers.png"), 0)
    c = cv2.imread(str(tmp_path / "c" / "rivers.png"), 0)
    assert (b != c).any() and (b[:, ::-1] != c).mean() < 0.01
    with pytest.raises(ValueError):
        export_engraving(p, tmp_path / "d", lines="dotted")


def test_city_engraving_is_cut_to_the_window(registry, resolver, tmp_path):
    p = puzzles.build_puzzle(registry, resolver, "city", "Gridville", ("streets",), SMALL,
                             options=puzzles.PuzzleOptions(budget_km=60))
    m = export_engraving(p, tmp_path, width_mm=80, margin_mm=4, dpi=127)
    c = _coords((tmp_path / "streets.svg").read_text())
    assert c[:, 0].min() >= 3.9 and c[:, 0].max() <= 76.1 and m["target"]["kind"] == "city"
    contour = shapely.box(4, 4, 76, m["canvas_mm"][1] - 4)
    assert _coords((tmp_path / "contour.svg").read_text()).min(axis=0) == pytest.approx(contour.bounds[:2], abs=0.01)


def test_python_interface(store, tmp_path):
    gq = GeoQuiz(store.root.parent)
    assert gq.countries()["iso3"].tolist() == ["OTH", "SQL"] and gq.countries(continent="Asia").empty
    cities = gq.cities(country="squareland", min_population=1e5)
    assert cities["name"].tolist() == ["Gridville"] and cities["population"][0] == 120000
    f = gq.features("country", "SQL").set_index("feature")
    assert f.loc["rivers", "ready"] and f.loc["rivers", "source"] == "hydrosheds"
    assert not f.loc["rail", "ready"] and "ingest" in f.loc["rail", "get_it_with"]
    assert gq.features("city", "Gridville").set_index("feature").loc["streets", "ready"]
    assert "neon" in gq.styles() and gq.settings()["width_mm"] == 200.0
    m = gq.engrave("country", "SQL", features=["rivers", "lakes"], out=tmp_path, ingest_missing=False, width_mm=90, dpi=127)
    assert m["canvas_mm"][0] == 90 and (tmp_path / "SQL_rivers-lakes_engrave" / "all.svg").exists()
    meta = gq.render("country", "SQL", out=tmp_path, ingest_missing=False, size="180x320", fps=10, phases="1,0,1,0.5",
                     engrave=True, width_mm=60, dpi=127)
    assert (tmp_path / "SQL_rivers-lakes_neon" / "engrave" / "lakes.png").exists() and meta["engrave"]["canvas_mm"][0] == 60
    with pytest.raises(ValueError, match="unknown settings"):
        gq.render("country", "SQL", ingest_missing=False, colour="red")


def test_empty_store_says_what_to_do(tmp_path):
    from geoquiz.providers import default_registry
    from geoquiz.regions import RegionResolver
    from geoquiz.store import FeatureStore
    resolver = RegionResolver(default_registry(FeatureStore(tmp_path / "store")))
    with pytest.raises(ValueError, match="geoquiz ingest --sources natural_earth"):
        resolver.country("CHE")
    with pytest.raises(ValueError, match="geoquiz ingest --sources ghsl"):
        resolver.city("Zürich")


def test_all_of_a_continent_or_country(store, tmp_path, capsys):
    from geoquiz.cli import main
    gq = GeoQuiz(store.root.parent)
    assert gq.targets("country", "Europe")["target"].tolist() == ["OTH", "SQL"]            # largest first
    assert gq.targets("country", "Europe", min_population=1.5e6)["name"].tolist() == ["Otherland"]
    assert gq.targets("country", limit=1)["target"].tolist() == ["OTH"]
    cities = gq.targets("city", "Squareland")
    assert cities[["target", "country"]].values.tolist() == [["Gridville", "Squareland"]]
    with pytest.raises(ValueError, match="no continent"):
        gq.targets("country", "Atlantis")
    with pytest.raises(ValueError, match="no cities found"):
        gq.targets("city", "Otherland")
    # same settings for every place; one that lacks the data is skipped and reported
    res = gq.make_all("country", "Europe", video=False, engrave=True, features=["rivers", "lakes", "coast"],
                      out=tmp_path / "all", ingest_missing=False, width_mm=70, dpi=127)
    assert [d["name"] for d in res["done"]] == ["Squareland"] and [f["name"] for f in res["failed"]] == ["Otherland"]
    assert (tmp_path / "all" / "0002_SQL_rivers-lakes_engrave" / "all.svg").exists()
    # in a run over many places a missing feature is skipped rather than failing the place
    res = gq.make_all("city", "Squareland", video=False, engrave=True, features=["streets", "buildings"],
                      out=tmp_path / "cities", ingest_missing=False, budget_km=60, width_mm=70, dpi=127)
    assert list(res["done"][0]["result"]["layers"]) == ["streets"]
    capsys.readouterr()
    assert main(["--data", str(store.root.parent), "all", "country", "Europe", "--dry-run"]) == 0
    assert capsys.readouterr().out.strip() == "2 countries: Otherland, Squareland"
