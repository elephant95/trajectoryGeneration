"""Small non-geographic assets fetched at ingest: country flags (flag-icons, MIT)."""
from __future__ import annotations

import tarfile
from pathlib import Path

from .ingest import RawCache

NPM = "https://registry.npmjs.org/flag-icons"


def ingest_flags(raw: RawCache, refresh: bool = False) -> Path:
    """Download the flag-icons package from npm and unpack its 4x3 SVGs."""
    meta = raw.get_json("flag_icons", f"{NPM}/latest", "latest.json", refresh=refresh)
    tgz = raw.fetch("flag_icons", meta["dist"]["tarball"], refresh=refresh)
    out = raw.root / "flag_icons" / "4x3"
    if not out.exists() or refresh:
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as tar:
            for m in tar.getmembers():
                if m.isfile() and "/flags/4x3/" in m.name and m.name.endswith(".svg"):
                    (out / Path(m.name).name).write_bytes(tar.extractfile(m).read())
    return out


def flag_svg(raw_root: Path, iso_a2: str | None) -> bytes | None:
    """SVG bytes for a country's flag, or None when flags were not ingested."""
    if not iso_a2:
        return None
    p = Path(raw_root) / "flag_icons" / "4x3" / f"{iso_a2.lower()}.svg"
    return p.read_bytes() if p.exists() else None
