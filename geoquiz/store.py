"""GeoParquet feature store.

Layout: ``<root>/<source>/<feature>/<partition>.lod<N>.parquet``. A partition
is ``global`` or a source-specific region (``eu`` for HydroRIVERS Europe,
``switzerland`` for a Geofabrik extract, ``CHE-ADM1`` for geoBoundaries).

Files are written with a bbox covering column and Hilbert-sorted row groups,
so a read by bounding box only touches the row groups it needs.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from . import schema
from .schema import LOD, Feature


class FeatureStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, source: str, feature: Feature, partition: str = "global", lod: LOD = LOD.FULL) -> Path:
        return self.root / source / Feature(feature).value / f"{partition}.lod{int(lod)}.parquet"

    def has(self, source: str, feature: Feature, partition: str = "global", lod: LOD = LOD.FULL) -> bool:
        return self.path(source, feature, partition, lod).exists()

    def partitions(self, source: str, feature: Feature) -> list[str]:
        d = self.root / source / Feature(feature).value
        return sorted({p.name.split(".lod")[0] for p in d.glob("*.lod0.parquet")}) if d.exists() else []

    def write(self, gdf: gpd.GeoDataFrame, source: str, feature: Feature, partition: str = "global",
              lods: tuple[LOD, ...] = (LOD.FULL, LOD.MEDIUM, LOD.COARSE)) -> list[Path]:
        """Write ``gdf`` at every requested level of detail."""
        schema.validate(gdf)
        if len(gdf) > 1:
            gdf = gdf.iloc[gdf.geometry.hilbert_distance().argsort()].reset_index(drop=True)
        written = []
        for lod in lods:
            g = simplify(gdf, lod)
            p = self.path(source, feature, partition, lod)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            g.to_parquet(tmp, index=False, write_covering_bbox=True, schema_version="1.1.0",
                         row_group_size=20_000, compression="zstd")
            tmp.replace(p)
            written.append(p)
        return written

    def read(self, source: str, feature: Feature, partition: str | list[str] | None = None, lod: LOD = LOD.FULL,
             bbox: tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
        """Read one partition, a list of partitions, or every partition when ``partition`` is None.

        ``bbox`` is (west, south, east, north) in degrees. Falls back to the
        nearest finer LOD when the requested one was not written.
        """
        parts = ([partition] if isinstance(partition, str) else list(partition) if partition is not None
                 else self.partitions(source, feature))
        frames = []
        for part in parts:
            p = next((self.path(source, feature, part, LOD(l)) for l in range(int(lod), -1, -1)
                      if self.has(source, feature, part, LOD(l))), None)
            if p is None:
                continue
            frames.append(gpd.read_parquet(p, bbox=bbox))
        if not frames:
            return schema.empty()
        out = frames[0] if len(frames) == 1 else gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=schema.CRS)
        if bbox is not None and len(out):
            # the covering-bbox filter is per feature bbox; make the test exact
            out = out[out.geometry.intersects(shapely.box(*bbox))].reset_index(drop=True)
        return out

    def inventory(self) -> pd.DataFrame:
        """One row per stored file: source, feature, partition, lod, rows, bytes."""
        import pyarrow.parquet as pq
        rows = []
        for p in sorted(self.root.glob("*/*/*.lod*.parquet")):
            part, lod = p.name[: -len(".parquet")].rsplit(".lod", 1)
            rows.append({"source": p.parts[-3], "feature": p.parts[-2], "partition": part, "lod": int(lod),
                         "rows": pq.ParquetFile(p).metadata.num_rows, "bytes": p.stat().st_size})
        return pd.DataFrame(rows)


def simplify(gdf: gpd.GeoDataFrame, lod: LOD) -> gpd.GeoDataFrame:
    """Topology-preserving simplification to an LOD; drops features that vanish."""
    tol = schema.LOD_TOLERANCE_DEG[LOD(lod)]
    if tol == 0 or not len(gdf):
        return gdf
    g = gdf.copy()
    g["geometry"] = g.geometry.simplify(tol, preserve_topology=True)
    polys = g.geom_type.isin(["Polygon", "MultiPolygon"])
    if polys.any():
        # a polygon smaller than the tolerance cell is invisible at this LOD
        keep = ~polys | (shapely.area(g.geometry.to_numpy()) > tol * tol)
        g = g[keep]
    return g[~g.geometry.is_empty].reset_index(drop=True)
