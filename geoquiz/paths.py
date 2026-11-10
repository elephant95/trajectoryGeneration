"""Where data lives on disk. Everything hangs off one root directory."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @classmethod
    def default(cls) -> "DataPaths":
        return cls(Path(os.environ.get("GEOQUIZ_DATA", "geoquiz_data")).resolve())

    @property
    def raw(self) -> Path:
        """Downloaded source files, untouched, plus manifest.json."""
        return self.root / "raw"

    @property
    def store(self) -> Path:
        """Normalised GeoParquet feature store."""
        return self.root / "store"

    @property
    def out(self) -> Path:
        """Finished puzzles, one folder each."""
        return self.root / "out"
