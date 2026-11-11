"""Download raw sources once, with checksums and version stamps.

Nothing here runs at render time: ingest fills ``raw/``, the provider
adapters normalise ``raw/`` into the store, and rendering reads only the
store. ``raw/manifest.json`` records what was fetched, from where and when,
which is the data-version trail each puzzle's metadata points back to.
"""
from __future__ import annotations

import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

USER_AGENT = "geoquiz/0.1 (offline batch ingest)"
RETRIES = 4          # per address, with 1, 2, 4, 8 s pauses


class DownloadError(ConnectionError):
    """A source could not be downloaded from any of its addresses."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RawCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    def _manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def record(self, key: str) -> dict | None:
        return self._manifest().get(key)

    def fetch(self, source: str, url: str, filename: str | None = None, *, sha256: str | None = None,
              refresh: bool = False, mirrors: tuple[str, ...] = ()) -> Path:
        """Download ``url`` to ``raw/<source>/<filename>`` unless already there.

        A file that is present but missing from the manifest (for example one
        copied in by hand) is hashed and recorded rather than re-downloaded.
        A failed connection is retried, then each of ``mirrors`` is tried.
        """
        filename = filename or url.rsplit("/", 1)[-1]
        dest = self.root / source / filename
        key = f"{source}/{filename}"
        man = self._manifest()
        if dest.exists() and not refresh:
            if key not in man:
                man[key] = self._entry(dest, url, fetched=False)
                self._save(man)
            self._check(dest, man[key]["sha256"], sha256)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        last_modified, used = self._download([url, *mirrors], tmp)
        url = used
        tmp.replace(dest)
        man = self._manifest()
        man[key] = self._entry(dest, url, fetched=True, last_modified=last_modified)
        self._save(man)
        self._check(dest, man[key]["sha256"], sha256)
        return dest

    @staticmethod
    def _download(urls: list[str], tmp: Path) -> tuple[str | None, str]:
        """Stream the first URL that works into ``tmp``; returns (Last-Modified, the URL used)."""
        errors = []
        for url in urls:
            for attempt in range(RETRIES):
                try:
                    with requests.get(url, stream=True, timeout=(20, 120), headers={"User-Agent": USER_AGENT}) as r:
                        r.raise_for_status()
                        with tmp.open("wb") as f:
                            for chunk in r.iter_content(1 << 20):
                                f.write(chunk)
                        return r.headers.get("Last-Modified"), url
                except requests.HTTPError as e:              # the server answered: retrying will not help
                    errors.append(f"{url}: {e}")
                    break
                except (requests.ConnectionError, requests.Timeout) as e:
                    errors.append(f"{url}: {type(e).__name__}")
                    time.sleep(2 ** attempt)
        tmp.unlink(missing_ok=True)
        host = urls[0].split("/")[2]
        raise DownloadError(
            f"could not download {urls[0].rsplit('/', 1)[-1]} (tried {len(errors)} times across {len(urls)} address(es); "
            f"last: {errors[-1]}). Check the internet connection, VPN or firewall for {host}, then run the same command "
            "again: files already downloaded are kept and skipped.")

    def get_json(self, source: str, url: str, filename: str, refresh: bool = False) -> dict:
        return json.loads(self.fetch(source, url, filename, refresh=refresh).read_text())

    def extract(self, archive: Path, members: list[str] | None = None) -> Path:
        """Unzip next to the archive (into ``<stem>/``); returns the directory."""
        out = archive.with_suffix("")
        marker = out / ".extracted"
        if marker.exists():
            return out
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(out, members)
        marker.touch()
        return out

    def _entry(self, path: Path, url: str, fetched: bool, last_modified: str | None = None) -> dict:
        return {
            "url": url,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "last_modified": last_modified,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "downloaded_by_ingest": fetched,
        }

    def _save(self, man: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(man, indent=2, sort_keys=True))

    @staticmethod
    def _check(path: Path, actual: str, expected: str | None) -> None:
        if expected and actual != expected:
            raise ValueError(f"checksum mismatch for {path}: {actual} != {expected}")
