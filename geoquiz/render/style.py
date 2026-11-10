"""Visual styles. One style for now (neon glow on dark); a style is only data."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schema import Feature

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Style:
    name: str = "neon"
    background: RGB = (0.020, 0.030, 0.055)
    background_centre: RGB = (0.045, 0.065, 0.110)     # subtle radial lift behind the map
    text: RGB = (0.96, 0.97, 1.0)
    text_muted: RGB = (0.62, 0.68, 0.78)
    #: one accent hue per feature class
    colors: dict = field(default_factory=lambda: {
        Feature.RIVER: (0.30, 0.74, 1.00),
        Feature.LAKE: (0.22, 0.60, 1.00),
        Feature.WATERWAY: (0.25, 0.62, 1.00),
        Feature.WATER: (0.16, 0.45, 0.95),
        Feature.STREET: (1.00, 0.70, 0.28),
        Feature.RAIL: (1.00, 0.36, 0.48),
        Feature.COUNTRY: (1.00, 1.00, 1.00),
    })
    glow_near: float = 1.5        # tight halo (quarter-resolution blur)
    glow_far: float = 1.1         # wide bloom (sixteenth-resolution blur)
    head: float = 0.7             # white-hot tip where lines are being drawn
    fade: float = 0.02            # reveal softness, as a fraction of the reveal
    outside_dim: float = 0.25     # brightness of features outside the target (country mode)
    font_display: str = "Inter Display"
    font_text: str = "Inter"

    def color(self, feature: Feature) -> RGB:
        return self.colors.get(Feature(feature), self.text)


NEON = Style()
STYLES = {"neon": NEON}
