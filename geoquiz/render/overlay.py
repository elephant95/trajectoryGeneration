"""Text and graphics drawn over the map: question, countdown, answer card, credit.

Layout (for a 1080 x 1920 frame, scaled for other sizes): the question sits in
the top third, the map in the middle band, the countdown low in the centre,
and the bottom fifth stays free of essentials because Instagram puts its own
caption and buttons there. The credit sits at the lower-left corner of the map.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
import skia

from ..scene import Frame
from .style import Style


@dataclass(frozen=True)
class Timeline:
    """Phase boundaries in seconds."""
    duration: float = 15.0
    reveal_end: float = 8.0       # 0..reveal_end: the network grows (the question is visible from frame 0)
    countdown_end: float = 11.0   # 3-2-1
    answer_end: float = 14.0      # answer card; then the outro fades back to the opening frame

    @classmethod
    def for_duration(cls, duration: float) -> "Timeline":
        k = duration / 15.0
        return cls(duration, 8.0 * k, 11.0 * k, 14.0 * k)

    @classmethod
    def from_phases(cls, reveal: float = 8.0, countdown: float = 3.0, answer: float = 3.0, outro: float = 1.0) -> "Timeline":
        """Build from the length of each phase in seconds. ``countdown=0`` skips the countdown."""
        if min(reveal, answer, outro) <= 0 or countdown < 0:
            raise ValueError("reveal, answer and outro must be positive, countdown non-negative")
        return cls(reveal + countdown + answer + outro, reveal, reveal + countdown, reveal + countdown + answer)


@dataclass(frozen=True)
class Texts:
    question: tuple[str, ...]     # one or two lines
    hint: str
    answer: str
    fact: str
    credit: str


#: Vertical band of the frame that holds the map, as fractions of the frame height.
MAP_BAND = (530 / 1920, 1400 / 1920)


def map_box(frame: Frame, margin: float = 0.0) -> tuple[float, float, float, float]:
    """Pixel box (left, top, right, bottom) the map occupies."""
    m = margin * frame.width
    return (m, MAP_BAND[0] * frame.height, frame.width - m, MAP_BAND[1] * frame.height)


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _c4(rgb, a: float = 1.0) -> skia.Color4f:
    return skia.Color4f(rgb[0], rgb[1], rgb[2], a)


class Overlay:
    def __init__(self, frame: Frame, style: Style, texts: Texts, timeline: Timeline,
                 flag_svg: bytes | None = None, globe: skia.Image | None = None, accent=(1.0, 1.0, 1.0)):
        self.frame, self.style, self.texts, self.tl = frame, style, texts, timeline
        self.u = frame.width / 1080.0
        self.accent = accent
        self.globe = globe
        self.flag = self._flag_image(flag_svg) if flag_svg else None
        self.display = skia.Typeface(style.font_display, skia.FontStyle.Bold())
        self.text_face = skia.Typeface(style.font_text, skia.FontStyle.Normal())
        self.text_bold = skia.Typeface(style.font_text, skia.FontStyle(600, 5, skia.FontStyle.kUpright_Slant))

    # -- helpers --------------------------------------------------------------
    def _font(self, face, size: float) -> skia.Font:
        f = skia.Font(face, size * self.u)
        f.setSubpixel(True)
        f.setEdging(skia.Font.Edging.kAntiAlias)
        return f

    def _fit(self, face, text: str, size: float, max_width: float) -> skia.Font:
        f = self._font(face, size)
        w = f.measureText(text)
        if w > max_width * self.u:
            f = self._font(face, size * max_width * self.u / w)
        return f

    def _centered(self, canvas, text: str, font: skia.Font, y: float, color, alpha: float = 1.0, dy: float = 0.0):
        if alpha <= 0 or not text:
            return
        w = font.measureText(text)
        canvas.drawString(text, (self.frame.width - w) / 2, (y + dy) * self.u, font,
                          skia.Paint(AntiAlias=True, Color4f=_c4(color, alpha)))

    def _flag_image(self, svg: bytes, w: int = 132, h: int = 99) -> skia.Image | None:
        dom = skia.SVGDOM.MakeFromStream(skia.MemoryStream(svg))
        if dom is None:
            return None
        ss = 2
        pw, ph = int(w * self.u * ss), int(h * self.u * ss)
        surface = skia.Surface(pw, ph)
        c = surface.getCanvas()
        c.clear(skia.ColorTRANSPARENT)
        c.clipRRect(skia.RRect.MakeRectXY(skia.Rect.MakeWH(pw, ph), pw * 0.07, pw * 0.07), True)
        dom.setContainerSize(skia.Size(640, 480))
        c.scale(pw / 640, ph / 480)
        dom.render(c)
        return surface.makeImageSnapshot()

    def _image(self, canvas, image: skia.Image, cx: float, cy: float, w: float, h: float, alpha: float, scale: float = 1.0):
        if alpha <= 0:
            return
        w, h = w * self.u * scale, h * self.u * scale
        dst = skia.Rect.MakeXYWH(cx * self.u - w / 2, cy * self.u - h / 2, w, h)
        paint = skia.Paint(AntiAlias=True, Alphaf=alpha)
        canvas.drawImageRect(image, dst, skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear), paint)

    # -- phases ---------------------------------------------------------------
    def draw(self, canvas: skia.Canvas, t: float) -> None:
        tl, st = self.tl, self.style
        outro = (t - tl.answer_end) / max(tl.duration - tl.answer_end, 1e-6)       # 0..1 in the outro
        # question: on from the first frame, out when the answer lands, back for the loop
        q = 1 - _smooth((t - tl.countdown_end) / 0.3)
        if outro > 0:
            q = _smooth((outro - 0.45) / 0.55)
        if q > 0:
            font = self._font(self.display, 92)
            for i, line in enumerate(self.texts.question):
                self._centered(canvas, line, font, 318 + i * 104, st.text, q)
            base = 318 + (len(self.texts.question) - 1) * 104
            self._centered(canvas, self.texts.hint, self._fit(self.text_face, self.texts.hint, 38, 960), base + 78,
                           self.accent, q * 0.95)

        # credit: on screen whenever the map is
        credit = self._fit(self.text_face, self.texts.credit, 21, 990)
        canvas.drawString(self.texts.credit, 44 * self.u, MAP_BAND[1] * self.frame.height + 4 * self.u, credit,
                          skia.Paint(AntiAlias=True, Color4f=_c4(st.text_muted, 0.85)))

        # countdown
        if tl.reveal_end <= t < tl.countdown_end:
            self._countdown(canvas, t)

        # answer card
        ta = t - tl.countdown_end
        if ta >= 0:
            out = 1 - _smooth(outro / 0.4) if outro > 0 else 1.0
            pop = _smooth(ta / 0.3)
            have_flag, have_globe = self.flag is not None, self.globe is not None
            if have_flag:
                cx = 540 - (75 if have_globe else 0)
                self._image(canvas, self.flag, cx, 262, 132, 99, pop * out, 0.85 + 0.15 * pop)
            if have_globe:
                cx = 540 + (82 if have_flag else 0)
                g = _smooth((ta - 0.15) / 0.35)
                self._image(canvas, self.globe, cx, 262, 118, 118, g * out, 0.85 + 0.15 * g)
            name = self.texts.answer
            font = self._fit(self.display, name, 118, 980)
            shown = name[: int(np.ceil(len(name) * min(max((ta - 0.15) / 0.55, 0.0), 1.0)))]
            if shown and out > 0:                                   # typed in, anchored to the full name's position
                x = (self.frame.width - font.measureText(name)) / 2
                canvas.drawString(shown, x, 448 * self.u, font, skia.Paint(AntiAlias=True, Color4f=_c4(st.text, out)))
            f = _smooth((ta - 0.8) / 0.4) * out
            self._centered(canvas, self.texts.fact, self._fit(self.text_face, self.texts.fact, 36, 980), 508,
                           st.text_muted, f, dy=(1 - _smooth((ta - 0.8) / 0.4)) * 10)

    def _countdown(self, canvas: skia.Canvas, t: float) -> None:
        tl, st, u = self.tl, self.style, self.u
        total = tl.countdown_end - tl.reveal_end
        step = total / 3
        k = min(int((t - tl.reveal_end) / step), 2)
        frac = (t - tl.reveal_end - k * step) / step
        cx, cy, r = self.frame.width / 2, 1484 * u, 54 * u
        a = min(_smooth((t - tl.reveal_end) / 0.2), _smooth((tl.countdown_end - t) / 0.15))
        ring = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=7 * u, StrokeCap=skia.Paint.kRound_Cap)
        ring.setColor4f(_c4(st.text_muted, 0.25 * a))
        canvas.drawCircle(cx, cy, r, ring)
        ring.setColor4f(_c4(self.accent, a))
        canvas.drawArc(skia.Rect.MakeLTRB(cx - r, cy - r, cx + r, cy + r), -90, 360 * (1 - frac), False, ring)
        digit = str(3 - k)
        s = 1.0 + 0.25 * (1 - _smooth(frac / 0.25))                  # each digit pops in
        font = self._font(self.display, 62 * s)
        w = font.measureText(digit)
        canvas.drawString(digit, cx - w / 2, cy + 22 * u * s, font, skia.Paint(AntiAlias=True, Color4f=_c4(st.text, a)))


def render_globe(land, lon: float, lat: float, size: int, style: Style, accent=(1.0, 1.0, 1.0)) -> skia.Image:
    """A small locator globe: orthographic view centred on the target, with a marker.

    ``land`` is the small-scale land layer (Natural Earth 1:110m).
    """
    ss = 2
    n = size * ss
    surface = skia.Surface(n, n)
    c = surface.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    r = n / 2 - 3 * ss
    centre = n / 2
    c.drawCircle(centre, centre, r, skia.Paint(AntiAlias=True, Color4f=skia.Color4f(*style.globe_ocean, 1)))
    lam0, phi0 = np.radians(lon), np.radians(lat)
    path = skia.Path()
    geoms = shapely.segmentize(land.geometry.to_numpy(), 1.5)
    for poly in shapely.get_parts(geoms):
        ll = np.radians(shapely.get_coordinates(poly.exterior))
        lam, phi = ll[:, 0] - lam0, ll[:, 1]
        x = np.cos(phi) * np.sin(lam)
        y = np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(lam)
        z = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(lam)
        if (z < 0).all():
            continue
        back = z < 0                          # far side: slide to the limb so the fill closes along the horizon
        norm = np.maximum(np.hypot(x, y), 1e-9)
        x, y = np.where(back, x / norm, x), np.where(back, y / norm, y)
        pts = np.column_stack([centre + x * r, centre - y * r]).tolist()
        path.moveTo(*pts[0])
        for p in pts[1:]:
            path.lineTo(*p)
        path.close()
    c.save()
    clip = skia.Path()
    clip.addCircle(centre, centre, r)
    c.clipPath(clip, True)
    c.drawPath(path, skia.Paint(AntiAlias=True, Color4f=skia.Color4f(*style.globe_land, 1)))
    c.restore()
    c.drawCircle(centre, centre, r, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.5 * ss,
                                               Color4f=skia.Color4f(*style.text_muted, 0.7)))
    c.drawCircle(centre, centre, r * 0.20, skia.Paint(AntiAlias=True, Color4f=skia.Color4f(*accent, 0.30)))
    c.drawCircle(centre, centre, r * 0.085, skia.Paint(AntiAlias=True, Color4f=skia.Color4f(*accent, 1)))
    return surface.makeImageSnapshot()
