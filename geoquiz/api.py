"""Python interface: everything the command line does, as method calls.

    from geoquiz import GeoQuiz
    gq = GeoQuiz()                                   # uses ./geoquiz_data
    gq.countries(continent="Europe")                 # DataFrame: iso3, name, continent, population
    gq.cities(country="Italy", min_population=2e5)   # DataFrame: name, country, population
    gq.features("country", "Austria")                # what can be drawn there, and what is ready
    gq.render("country", "Brazil")                   # -> metadata dict; video in geoquiz_data/out/…
    gq.engrave("country", "CHE", features=["lakes", "roads", "rail"], width_mm=200)
    gq.make_all("country", "South America")          # one Reel per country, same settings
    gq.make_all("city", "Brazil", min_population=1e6, video=False, engrave=True)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import cli
from .regions import fold


class GeoQuiz:
    def __init__(self, data: str | Path | None = None):
        """``data`` is the data root; default ``$GEOQUIZ_DATA`` or ``./geoquiz_data``."""
        self._ctx = cli._context(argparse.Namespace(data=str(data) if data else None))
        self.paths, self.store, self.raw, self.registry, self.resolver = self._ctx

    def _basics(self, cities: bool = False) -> None:
        from .auto import ensure_basics
        ensure_basics(self.registry, self.raw, cities=cities)

    # -- what exists ------------------------------------------------------------------
    def countries(self, continent: str | None = None) -> pd.DataFrame:
        """Every country that can be a target. Downloads the country list on first use."""
        self._basics()
        t = self.registry.get("natural_earth").countries()
        if continent:
            t = t[t["x_continent"].map(fold) == fold(continent)]
        return pd.DataFrame({"iso3": t["source_id"], "name": t["name"], "continent": t["x_continent"],
                             "population": t["x_pop"]}).sort_values("name").reset_index(drop=True)

    def cities(self, country: str | None = None, min_population: float = 0) -> pd.DataFrame:
        """Every city that can be a target (GHSL urban centres), largest first.
        Downloads the city list (about 310 MB) on first use."""
        self._basics(cities=True)
        t = self.registry.get("ghsl").urban_centres()
        if country:
            t = t[t["x_country"].map(fold) == fold(country)]
        t = t[t["x_pop"] >= min_population]
        return pd.DataFrame({"name": t["name"], "country": t["x_country"], "population": t["x_pop"].round(),
                             "also_contains": t["x_names_all"]}).sort_values("population", ascending=False).reset_index(drop=True)

    def features(self, kind: str, target: str, country: str | None = None) -> pd.DataFrame:
        """The features this place supports: whether each is in the store, and how to get it if not."""
        self._basics(cities=kind == "city")
        st = cli.feature_status(self._ctx, kind, target, country)["features"]
        return pd.DataFrame([{"feature": k, **v} for k, v in st.items()])

    @staticmethod
    def styles() -> list[str]:
        from .render.style import STYLES
        return list(STYLES)

    @staticmethod
    def settings() -> dict:
        """Every setting ``render`` and ``engrave`` accept, with its default."""
        return dict(cli.RENDER_DEFAULTS)

    # -- making things ----------------------------------------------------------------------
    def render(self, kind: str, target: str, features: list[str] | None = None, out: str | Path | None = None,
               ingest_missing: bool = True, **settings) -> dict:
        """Render one Reel. ``settings`` are the keys of :meth:`settings` (style, outside, reveal,
        question, duration, engrave=True, …). Returns the metadata, including ``folder``."""
        s = {"features": features, "ingest_missing": ingest_missing, **settings}
        return cli.render_one(self._ctx, kind, target, {k: v for k, v in s.items() if v is not None},
                              Path(out) if out else self.paths.out)

    def engrave(self, kind: str, target: str, features: list[str] | None = None, out: str | Path | None = None,
                ingest_missing: bool = True, **settings) -> dict:
        """Laser-engraving files only: one SVG and PNG per feature, the contour, and all together."""
        s = {"features": features, "ingest_missing": ingest_missing, **settings}
        return cli.engrave_one(self._ctx, kind, target, {k: v for k, v in s.items() if v is not None},
                               Path(out) if out else self.paths.out)

    def targets(self, kind: str, where: str | None = None, min_population: float = 0, limit: int | None = None) -> pd.DataFrame:
        """Every country of a continent, or every city of a country, largest first: what :meth:`make_all` would make."""
        return pd.DataFrame(cli.select_targets(self._ctx, kind, where, min_population, limit))

    def make_all(self, kind: str, where: str | None = None, min_population: float = 0, limit: int | None = None,
                 video: bool = True, engrave: bool = False, features: list[str] | None = None,
                 out: str | Path | None = None, ingest_missing: bool = True, **settings) -> dict:
        """The same video and/or engraving files for every country of a continent
        (``make_all("country", "South America")``) or every city of a country
        (``make_all("city", "Brazil", min_population=1e6)``). Places that fail are skipped and listed
        under ``failed`` in the returned dict."""
        from .auto import partition_name
        targets = cli.select_targets(self._ctx, kind, where, min_population, limit)
        s = {"features": features, "ingest_missing": ingest_missing, **settings}
        folder = Path(out) if out else self.paths.out / f"all_{partition_name(where or 'world')}_{kind}"
        return cli.run_all(self._ctx, kind, targets, {k: v for k, v in s.items() if v is not None}, folder,
                           video=video, engrave=engrave)

    def batch(self, config: str | Path, out: str | Path | None = None) -> int:
        """Run a YAML batch file; returns 0 when every puzzle rendered."""
        return cli.cmd_batch(argparse.Namespace(data=str(self.paths.root), config=str(config), out=str(out) if out else None))

    def ingest(self, sources: str = "natural_earth,hydrosheds,ghsl,flags", **options) -> None:
        """Download and normalise sources, as ``geoquiz ingest`` does (hydro_regions="all", city="Paris", …)."""
        ns = dict(data=str(self.paths.root), sources=sources, hydro_regions="eu", country=None, adm="ADM0,ADM1",
                  osm_extract=None, osm_pbf=None, osm_themes="streets,rail,waterways,water", city=None,
                  in_country=None, overture_themes=None, refresh=False)
        unknown = set(options) - set(ns)
        if unknown:
            raise ValueError(f"unknown ingest options {sorted(unknown)}")
        cli.cmd_ingest(argparse.Namespace(**{**ns, **options}))
        self.registry.forget_coverage()
