"""The common schema every provider returns.

Renderers and planners only ever see these columns, never a source's own
field names. Source-specific extras are allowed as columns prefixed ``x_``
(for example ``x_dist_dn_km`` from HydroRIVERS, used by the reveal).
"""
from __future__ import annotations

from enum import Enum, IntEnum

import geopandas as gpd
import numpy as np
import pandas as pd


class Feature(str, Enum):
    RIVER = "river"            # lines
    LAKE = "lake"              # polygons
    COASTLINE = "coastline"    # lines
    LAND = "land"              # polygons, small-scale (locator globe)
    COUNTRY = "country"        # polygons, admin-0
    ADMIN = "admin"            # polygons, admin-1..5
    URBAN_CENTRE = "urban_centre"  # polygons, one per city
    STREET = "street"          # lines
    RAIL = "rail"              # lines
    WATERWAY = "waterway"      # lines, city-scale
    WATER = "water"            # polygons, city-scale
    BUILDING = "building"      # polygons


class Scale(str, Enum):
    COUNTRY = "country"
    CITY = "city"


class LOD(IntEnum):
    """Pre-simplified levels of detail held in the store."""
    FULL = 0      # source geometry
    MEDIUM = 1    # ~50 m tolerance
    COARSE = 2    # ~500 m tolerance


#: Simplification tolerance per LOD, in degrees (store geometry is WGS84).
LOD_TOLERANCE_DEG = {LOD.FULL: 0.0, LOD.MEDIUM: 0.00045, LOD.COARSE: 0.0045}

COLUMNS = ["geometry", "feature", "rank", "name", "source", "source_id",
           "licence", "attribution", "data_version"]

CRS = "EPSG:4326"


def empty() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({c: [] for c in COLUMNS if c != "geometry"}, geometry=[], crs=CRS)


def normalise(
    gdf: gpd.GeoDataFrame,
    *,
    feature: Feature,
    source: str,
    licence: str,
    attribution: str,
    data_version: str,
    rank=None,
    name=None,
    source_id=None,
    extras: dict | None = None,
) -> gpd.GeoDataFrame:
    """Coerce a source table into the common schema.

    ``rank``, ``name`` and ``source_id`` are array-likes aligned with ``gdf``
    (or scalars). ``rank`` is importance in 0..1 (river order, road class,
    population). ``extras`` maps ``x_*`` column names to array-likes.
    """
    if gdf.crs is None:
        raise ValueError(f"{source}: geometry has no CRS")
    geom = gdf.geometry.to_crs(CRS) if gdf.crs.to_epsg() != 4326 else gdf.geometry
    n = len(gdf)

    def col(v, default):
        if v is None:
            v = default
        if np.isscalar(v) or v is None:
            return np.full(n, v, dtype=object if isinstance(v, str) or v is None else None)
        return np.asarray(v)

    out = gpd.GeoDataFrame(
        {
            "feature": feature.value,
            "rank": np.clip(pd.to_numeric(pd.Series(col(rank, 0.5)), errors="coerce").fillna(0.0).to_numpy(float), 0, 1),
            "name": pd.Series(col(name, None), dtype="object").where(lambda s: s.notna(), None).to_numpy(),
            "source": source,
            "source_id": pd.Series(col(source_id, None)).astype("string").astype(object).to_numpy(),
            "licence": licence,
            "attribution": attribution,
            "data_version": data_version,
        },
        geometry=geom.to_numpy(),
        crs=CRS,
    )
    for k, v in (extras or {}).items():
        if not k.startswith("x_"):
            raise ValueError(f"extra column {k!r} must start with 'x_'")
        out[k] = np.asarray(v)
    out = out[~(out.geometry.is_empty | out.geometry.isna())].reset_index(drop=True)
    validate(out)
    return out


def validate(gdf: gpd.GeoDataFrame) -> None:
    """Raise if ``gdf`` does not follow the common schema."""
    missing = [c for c in COLUMNS if c not in gdf.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    stray = [c for c in gdf.columns if c not in COLUMNS and not c.startswith("x_")]
    if stray:
        raise ValueError(f"unexpected columns (extras must start with 'x_'): {stray}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError("geometry must be WGS84 (EPSG:4326)")
    if len(gdf) and not ((gdf["rank"] >= 0) & (gdf["rank"] <= 1)).all():
        raise ValueError("rank must lie in 0..1")
