"""GHSL Urban Centre Database R2024A: one worldwide definition of a city."""
from __future__ import annotations

import numpy as np
import pyogrio

from .. import licences, schema
from ..ingest import RawCache
from ..schema import LOD, Feature, Scale
from .base import StoreProvider

URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/"
       "GHS_UCDB_GLOBE_R2024A/V1-1/GHS_UCDB_GLOBE_R2024A_V1_1.zip")
GPKG = "GHS_UCDB_GLOBE_R2024A.gpkg"
LAYER = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"


def _clean(s):
    """The GeoPackage carries stray byte-order marks in field names and strings."""
    return s.replace("﻿", "").strip() if isinstance(s, str) else s


class GhslProvider(StoreProvider):
    name = "ghsl"
    licence = licences.GHSL_CC_BY
    features = {Feature.URBAN_CENTRE: {Scale.COUNTRY, Scale.CITY}}

    def ingest(self, raw: RawCache, refresh: bool = False) -> None:
        d = raw.extract(raw.fetch(self.name, URL, refresh=refresh), members=[GPKG])
        g = pyogrio.read_dataframe(d / GPKG, layer=LAYER)
        g.columns = [_clean(c) for c in g.columns]
        for c in ("GC_UCN_MAI_2025", "GC_UCN_LIS_2025", "GC_CNT_GAD_2025"):
            g[c] = g[c].map(_clean)
        pop = g["GC_POP_TOT_2025"].astype(float)
        out = schema.normalise(
            g, feature=Feature.URBAN_CENTRE, source=self.name, licence=self.name,
            attribution=self.licence.attribution, data_version="R2024A V1.1",
            name=g["GC_UCN_MAI_2025"], source_id=g["ID_UC_G0"],
            rank=(np.log10(pop.clip(lower=1)) - 4.7) / 2.8,        # 50 thousand -> 0, 30 million -> 1
            extras={"x_names_all": g["GC_UCN_LIS_2025"].to_numpy(),   # every named city in the centre
                    "x_country": g["GC_CNT_GAD_2025"].to_numpy(),
                    "x_pop": pop.to_numpy(),
                    "x_area_km2": g["GC_UCA_KM2_2025"].to_numpy(float)},
        )
        self.store.write(out, self.name, Feature.URBAN_CENTRE, lods=(LOD.FULL,))
        self.save_licence(self.name, self.licence)

    def urban_centres(self):
        return self.store.read(self.name, Feature.URBAN_CENTRE)
