"""Licence and attribution metadata attached to every layer.

Each provider declares a Licence; every row in the feature store carries the
licence id and attribution text, so a puzzle's credit line is generated from
the layers actually drawn rather than maintained by hand.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class Licence:
    id: str                 # SPDX id where one exists, otherwise a stable slug
    name: str
    url: str
    attribution: str        # short credit, shown on screen while the map is visible
    attribution_full: str   # full credit for the caption
    commercial_use: bool = True
    share_alike: bool = False        # derivative databases must be shared (ODbL)
    attribution_required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


NATURAL_EARTH = Licence(
    id="LicenseRef-NaturalEarth-PD",
    name="Public domain (Natural Earth)",
    url="https://www.naturalearthdata.com/about/terms-of-use/",
    attribution="Natural Earth",
    attribution_full="Made with Natural Earth (public domain).",
    attribution_required=False,
)

OSM_ODBL = Licence(
    id="ODbL-1.0",
    name="Open Database License 1.0",
    url="https://www.openstreetmap.org/copyright",
    attribution="© OpenStreetMap contributors",
    attribution_full="Map data © OpenStreetMap contributors, available under the Open Database License (openstreetmap.org/copyright).",
    share_alike=True,
)

HYDROSHEDS_V1 = Licence(
    id="LicenseRef-HydroSHEDS-v1",
    name="HydroSHEDS licence (v1 products)",
    url="https://www.hydrosheds.org/products",
    attribution="HydroSHEDS © WWF",
    attribution_full=(
        "This product incorporates data from the HydroSHEDS database which is "
        "© World Wildlife Fund, Inc. (2006-2013) and has been used herein under license."
    ),
)

GHSL_CC_BY = Licence(
    id="CC-BY-4.0",
    name="Creative Commons Attribution 4.0 (GHSL)",
    url="https://human-settlement.emergency.copernicus.eu/ghs_ucdb_2024.php",
    attribution="GHSL © European Union",
    attribution_full="GHS Urban Centre Database R2024A © European Union, CC BY 4.0.",
)

FLAG_ICONS_MIT = Licence(
    id="MIT",
    name="MIT (flag-icons)",
    url="https://github.com/lipis/flag-icons",
    attribution="flag-icons",
    attribution_full="Flags from flag-icons (MIT).",
    attribution_required=False,
)

_REGISTRY: dict[str, Licence] = {}


def register(licence: Licence, key: str | None = None) -> Licence:
    """Make a licence resolvable by key (rows in the store hold only the key)."""
    _REGISTRY[key or licence.id] = licence
    return licence


def get(key: str) -> Licence:
    return _REGISTRY[key]


def known() -> dict[str, Licence]:
    return dict(_REGISTRY)


for _l, _k in [(NATURAL_EARTH, "natural_earth"), (OSM_ODBL, "osm"), (HYDROSHEDS_V1, "hydrosheds"),
               (GHSL_CC_BY, "ghsl"), (FLAG_ICONS_MIT, "flag_icons")]:
    register(_l, _k)


def credit_line(attributions: Iterable[str], sep: str = " · ") -> str:
    """Short on-screen credit from the attribution texts of the drawn layers.

    Order is preserved and duplicates removed. OpenStreetMap is moved first
    because its credit is a licence obligation while the map is on screen.
    """
    seen: list[str] = []
    for a in attributions:
        if a and a not in seen:
            seen.append(a)
    seen.sort(key=lambda a: 0 if "OpenStreetMap" in a else 1)
    return sep.join(seen)
