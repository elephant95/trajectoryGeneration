"""geoBoundaries gbOpen: administrative subdivisions.

The release as a whole is CC BY 4.0, but each boundary keeps the licence of
its source (Switzerland's is swisstopo's), so the adapter reads the licence
from each boundary's metadata and stores it with the layer.
"""
from __future__ import annotations

import re

import pyogrio

from .. import schema
from ..ingest import RawCache
from ..licences import Licence
from ..schema import Feature, Scale
from .base import StoreProvider

API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{adm}/"


def licence_from_metadata(meta: dict) -> Licence:
    text = str(meta.get("boundaryLicense", "")).strip()
    source = str(meta.get("boundarySource", "")).strip()
    url = str(meta.get("licenseSource", "")).strip()
    low = text.lower()
    non_commercial = bool(re.search(r"non[- ]?commercial|\bnc\b", low))
    share_alike = bool(re.search(r"odbl|open database|share[- ]?alike|\bsa\b", low))
    return Licence(
        id=f"LicenseRef-geoBoundaries-{meta['boundaryID']}",
        name=text or "unknown",
        url=url if url.startswith("http") else f"https://{url}" if url else "https://www.geoboundaries.org",
        attribution=source or "geoBoundaries",
        attribution_full=f"Boundaries: {source} via geoBoundaries ({text}).",
        commercial_use=not non_commercial,
        share_alike=share_alike,
    )


def lfs_url(url: str) -> str:
    """geoBoundaries files are Git LFS objects; address the LFS media host directly."""
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/raw/(.+)", url)
    return f"https://media.githubusercontent.com/media/{m.group(1)}/{m.group(2)}" if m else url


class GeoBoundariesProvider(StoreProvider):
    name = "geoboundaries"
    licence = Licence(id="CC-BY-4.0", name="CC BY 4.0 (release); per-boundary source licences apply",
                      url="https://www.geoboundaries.org", attribution="geoBoundaries",
                      attribution_full="Boundaries from geoBoundaries (www.geoboundaries.org).")
    features = {Feature.COUNTRY: {Scale.COUNTRY}, Feature.ADMIN: {Scale.COUNTRY}}

    def ingest(self, raw: RawCache, iso3: str, levels: tuple[str, ...] = ("ADM0", "ADM1"), refresh: bool = False) -> dict:
        """Ingest boundary levels for one country; returns {level: Licence}."""
        found = {}
        for adm in levels:
            meta = raw.get_json(self.name, API.format(iso3=iso3, adm=adm), f"{iso3}-{adm}.json", refresh=refresh)
            lic = licence_from_metadata(meta)
            key = f"geoboundaries:{meta['boundaryID']}"
            path = raw.fetch(self.name, lfs_url(meta["gjDownloadURL"]), f"{iso3}-{adm}.geojson", refresh=refresh)
            g = pyogrio.read_dataframe(path)
            feature = Feature.COUNTRY if adm == "ADM0" else Feature.ADMIN
            out = schema.normalise(
                g, feature=feature, source=self.name, licence=key, attribution=lic.attribution,
                data_version=f"{meta['boundaryID']} ({meta.get('buildDate', 'n/a')})",
                name=g["shapeName"], rank=1.0 - int(adm[-1]) / 5,
                source_id=g["shapeID"] if adm != "ADM0" else iso3,
                extras={"x_level": [adm] * len(g), "x_iso3": [iso3] * len(g)},
            )
            self.store.write(out, self.name, feature, f"{iso3}-{adm}")
            self.save_licence(key, lic)
            found[adm] = lic
        return found
