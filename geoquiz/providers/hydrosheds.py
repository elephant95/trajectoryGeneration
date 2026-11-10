"""HydroSHEDS v1: HydroRIVERS (river reaches with order, discharge, distance to
outlet) and HydroLAKES. Dense enough for country-scale river puzzles, where
Natural Earth shows only the main stems."""
from __future__ import annotations

import numpy as np
import pyogrio

from .. import licences, schema
from ..ingest import RawCache
from ..schema import Feature, Scale
from .base import StoreProvider

RIVERS_URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_{region}_shp.zip"
LAKES_URL = "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip"
#: HydroRIVERS regional tiles: Africa, Arctic, Asia, Australasia, Europe, Greenland,
#: North America, South America, Siberia.
REGIONS = ("af", "ar", "as", "au", "eu", "gr", "na", "sa", "si")


class HydroShedsProvider(StoreProvider):
    name = "hydrosheds"
    licence = licences.HYDROSHEDS_V1
    features = {Feature.RIVER: {Scale.COUNTRY}, Feature.LAKE: {Scale.COUNTRY, Scale.CITY}}

    def ingest(self, raw: RawCache, regions: tuple[str, ...] = ("eu",), lakes: bool = True, refresh: bool = False) -> None:
        """Normalise HydroRIVERS for each region, and HydroLAKES clipped to the same extents."""
        common = dict(source=self.name, licence=self.name, attribution=self.licence.attribution, data_version="v1.0")
        lakes_shp = None
        if lakes:
            d = raw.extract(raw.fetch(self.name, LAKES_URL, refresh=refresh))
            lakes_shp = d / "HydroLAKES_polys_v10_shp" / "HydroLAKES_polys_v10.shp"
        for region in regions:
            if region not in REGIONS:
                raise ValueError(f"unknown HydroRIVERS region {region!r}; expected one of {REGIONS}")
            d = raw.extract(raw.fetch(self.name, RIVERS_URL.format(region=region), refresh=refresh))
            r = pyogrio.read_dataframe(d / f"HydroRIVERS_v10_{region}_shp" / f"HydroRIVERS_v10_{region}.shp")
            rivers = schema.normalise(
                r, feature=Feature.RIVER, **common, source_id=r["HYRIV_ID"],
                # 0.1 m³/s -> 0, 1000 m³/s -> 1 on a log scale
                rank=(np.log10(r["DIS_AV_CMS"].clip(lower=0.1)) + 1) / 4,
                extras={
                    "x_discharge_cms": r["DIS_AV_CMS"].to_numpy(float),
                    "x_strahler": r["ORD_STRA"].to_numpy(np.int16),
                    "x_length_km": r["LENGTH_KM"].to_numpy(float),
                    "x_dist_dn_km": r["DIST_DN_KM"].to_numpy(float),   # reach outlet to the sea or sink
                    "x_dist_up_km": r["DIST_UP_KM"].to_numpy(float),   # reach outlet to the furthest source
                    "x_main_riv": r["MAIN_RIV"].to_numpy(np.int64),    # id of the basin's most downstream reach
                    "x_next_down": r["NEXT_DOWN"].to_numpy(np.int64),
                },
            )
            bounds = tuple(rivers.total_bounds)
            del r
            self.store.write(rivers, self.name, Feature.RIVER, region)
            if lakes_shp is not None:
                l = pyogrio.read_dataframe(lakes_shp, bbox=bounds)
                lk = schema.normalise(
                    l, feature=Feature.LAKE, **common, source_id=l["Hylak_id"], name=l["Lake_name"],
                    rank=(np.log10(l["Lake_area"].clip(lower=0.1)) + 1) / 4,     # 0.1 km² -> 0, 1000 km² -> 1
                    extras={"x_area_km2": l["Lake_area"].to_numpy(float),
                            "x_type": l["Lake_type"].to_numpy(np.int8)},         # 1 lake, 2 reservoir, 3 lake control
                )
                self.store.write(lk, self.name, Feature.LAKE, region)
        self.save_licence(self.name, self.licence)
