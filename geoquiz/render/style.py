"""Visual styles. A style is only data: colours, glow, fonts, and whether ink
adds light to a dark background or takes it away from a light one."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..schema import Feature

RGB = tuple[float, float, float]

_NEON = {
    Feature.RIVER: (0.30, 0.74, 1.00),
    Feature.LAKE: (0.22, 0.60, 1.00),
    Feature.WATERWAY: (0.25, 0.62, 1.00),
    Feature.WATER: (0.16, 0.45, 0.95),
    Feature.COASTLINE: (0.55, 0.85, 1.00),
    Feature.STREET: (1.00, 0.70, 0.28),
    Feature.RAIL: (1.00, 0.36, 0.48),
    Feature.BUILDING: (0.95, 0.55, 0.75),
    Feature.ADMIN: (0.70, 0.95, 0.60),
    Feature.URBAN_CENTRE: (1.00, 0.86, 0.45),
    Feature.COUNTRY: (1.00, 1.00, 1.00),
}


@dataclass(frozen=True)
class Style:
    name: str = "neon"
    #: "add": ink is light on a dark background. "subtract": ink is pigment on a light background.
    mode: str = "add"
    background: RGB = (0.020, 0.030, 0.055)
    background_centre: RGB = (0.045, 0.065, 0.110)     # subtle lift behind the map
    text: RGB = (0.96, 0.97, 1.0)
    text_muted: RGB = (0.62, 0.68, 0.78)
    #: one accent hue per feature class
    colors: dict = field(default_factory=lambda: dict(_NEON))
    glow_near: float = 1.5        # tight halo (quarter-resolution blur)
    glow_far: float = 1.1         # wide bloom (sixteenth-resolution blur)
    head: float = 0.7             # extra intensity where lines are being drawn
    fade: float = 0.02            # reveal softness, as a fraction of the reveal
    outside_dim: float = 0.25     # brightness of features outside the target (outside="dim")
    globe_ocean: RGB = (0.07, 0.11, 0.19)
    globe_land: RGB = (0.30, 0.38, 0.52)
    font_display: str = "Inter Display"
    font_text: str = "Inter"

    def color(self, feature: Feature) -> RGB:
        return self.colors.get(Feature(feature), self.text)

    def accent(self, feature: Feature) -> RGB:
        """Colour for text and the countdown ring: the feature hue, readable on the background."""
        return self.color(feature)


NEON = Style()

INK = Style(
    name="ink", mode="subtract",
    background=(0.945, 0.925, 0.885), background_centre=(0.975, 0.960, 0.930),
    text=(0.10, 0.11, 0.14), text_muted=(0.38, 0.40, 0.45),
    colors={
        Feature.RIVER: (0.10, 0.30, 0.62), Feature.LAKE: (0.22, 0.45, 0.75), Feature.WATERWAY: (0.14, 0.36, 0.68),
        Feature.WATER: (0.30, 0.52, 0.80), Feature.COASTLINE: (0.10, 0.25, 0.50),
        Feature.STREET: (0.12, 0.12, 0.15), Feature.RAIL: (0.72, 0.16, 0.14), Feature.BUILDING: (0.25, 0.24, 0.26),
        Feature.ADMIN: (0.45, 0.30, 0.12), Feature.URBAN_CENTRE: (0.75, 0.22, 0.10),
        Feature.COUNTRY: (0.75, 0.10, 0.10),
    },
    glow_near=0.0, glow_far=0.12, head=0.25,
    globe_ocean=(0.80, 0.84, 0.88), globe_land=(0.45, 0.47, 0.50),
)

BLUEPRINT = Style(
    name="blueprint",
    background=(0.035, 0.150, 0.330), background_centre=(0.060, 0.215, 0.440),
    text=(0.95, 0.98, 1.0), text_muted=(0.68, 0.80, 0.94),
    colors={f: (0.90, 0.96, 1.00) for f in _NEON} | {
        Feature.LAKE: (0.62, 0.82, 1.00), Feature.WATER: (0.55, 0.76, 1.00), Feature.RAIL: (1.00, 0.86, 0.55),
        Feature.URBAN_CENTRE: (1.00, 0.90, 0.60), Feature.COUNTRY: (1.00, 0.92, 0.55)},
    glow_near=0.5, glow_far=0.35, head=0.4,
    globe_ocean=(0.03, 0.11, 0.26), globe_land=(0.45, 0.62, 0.86),
)

EMBER = Style(
    name="ember",
    background=(0.045, 0.020, 0.018), background_centre=(0.095, 0.040, 0.030),
    text=(1.0, 0.96, 0.92), text_muted=(0.78, 0.66, 0.60),
    colors={f: (1.00, 0.45, 0.15) for f in _NEON} | {
        Feature.RIVER: (1.00, 0.58, 0.20), Feature.LAKE: (1.00, 0.72, 0.35), Feature.WATER: (0.85, 0.30, 0.12),
        Feature.WATERWAY: (1.00, 0.60, 0.25), Feature.RAIL: (1.00, 0.86, 0.50), Feature.STREET: (1.00, 0.40, 0.14),
        Feature.URBAN_CENTRE: (1.00, 0.90, 0.60), Feature.COUNTRY: (1.00, 0.95, 0.85)},
    globe_ocean=(0.12, 0.06, 0.05), globe_land=(0.50, 0.30, 0.22),
)

MONO = Style(
    name="mono",
    background=(0.0, 0.0, 0.0), background_centre=(0.045, 0.045, 0.050),
    text=(1.0, 1.0, 1.0), text_muted=(0.65, 0.65, 0.68),
    colors={f: (0.94, 0.94, 0.96) for f in _NEON} | {Feature.WATER: (0.45, 0.45, 0.50), Feature.LAKE: (0.70, 0.70, 0.74)},
    glow_near=0.7, glow_far=0.5, head=0.5,
    globe_ocean=(0.08, 0.08, 0.09), globe_land=(0.42, 0.42, 0.45),
)

STYLES = {s.name: s for s in (NEON, INK, BLUEPRINT, EMBER, MONO)}


def with_overrides(style: Style, colors: dict | None = None, **fields) -> Style:
    """A copy of ``style`` with some colours or fields replaced (for config files)."""
    merged = dict(style.colors)
    for k, v in (colors or {}).items():
        merged[Feature(k)] = tuple(v)
    return replace(style, colors=merged, **fields)
