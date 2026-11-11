# geoquiz

Automated geo-quiz Reels: "which country is this, from its rivers?", "which
city is this, from its streets?". An offline Python pipeline on pre-downloaded
open data, rendered with Skia + NumPy/OpenCV and encoded with FFmpeg.

It covers phases 0 and 1 of the research report's roadmap (the data
foundation and the renderer) plus a general puzzle builder and a batch runner. It is independent of
`swiss_rail_rl` and shares only the repository.

## Install

```bash
pip install -e ".[geoquiz,dev]"     # from the repository root
```

Needs `ffmpeg` on the PATH (or `pip install imageio-ffmpeg`), and on a
headless Linux box `libegl1` for skia-python. Text uses the Inter and Inter
Display fonts when installed and falls back to the system default otherwise.

## Point and click

```bash
geoquiz gui          # or double-click geoquiz_gui.bat on Windows
```

This opens a page in your browser (served from your own computer, at
`http://127.0.0.1:8765`) where you pick the continent, the country or city,
what to draw, and whether to make a video, laser-engraving files or both.
Video and engraving settings (style, length, size in mm, thinnest line, …)
are form fields; progress and the results (a video player, a preview and
download link for every engraving layer, a button that opens the output
folder) appear on the same page. Nothing else needs a command.

Choosing **All countries of a continent** or **All cities of a country**
makes the same thing for every place with one click. A minimum population
and a maximum count (largest first) keep the run to a sensible size; the
page shows how many places match before you start, each finished place
appears as it completes, a place that fails is listed and skipped, and a
button stops the run after the current place.

## Run

```bash
geoquiz ingest                                 # Natural Earth, HydroSHEDS (Europe), GHSL, flags
geoquiz ingest --sources hydrosheds --hydro-regions all    # rivers and lakes for every continent (~2.5 GB stored)

geoquiz render country Brazil                               # rivers and lakes, shape hidden until the answer
geoquiz render country Austria --features roads,rail --style blueprint --ingest-missing
geoquiz render country Japan --features cities,coast --style ember
geoquiz render city Paris --features streets,water --style ink --ingest-missing
geoquiz render city Amsterdam --features buildings,water --radius-km 1.2 --ingest-missing

geoquiz all country "South America"                         # one Reel per country, same settings
geoquiz all city Brazil --min-pop 1000000 --limit 10 --engrave-only --width-mm 150
geoquiz all country Europe --dry-run                        # only list what would be made
geoquiz batch geoquiz/examples/batch.yaml      # many Reels from one config, including random picks
geoquiz list features | styles | countries | cities --in France
geoquiz resolve country Austria                # which features are ready there, and how to get the rest
geoquiz inventory                              # what the store holds, and the licences in play
pytest tests/geoquiz                           # offline, synthetic data
```

`--ingest-missing` downloads what a puzzle needs and the store lacks, once
per target; rendering itself only reads the local store.

## Inputs

**Target.** `country <name or ISO3>` (any Natural Earth country) or
`city <name>` (any of 11,422 GHSL urban centres, roughly every city above
50,000 people; add `--country` to disambiguate).

**Features** (`--features a,b,c`, the first is the primary one: it drives the
reveal, the accent colour and the fact).

| Mode | Feature | Source | Notes |
| --- | --- | --- | --- |
| country | `rivers`, `lakes` (default) | HydroRIVERS, HydroLAKES | thinned evenly across the frame to a stroke budget |
| country | `roads`, `rail` | Overture (or an OSM extract) | main road classes only; whole-country download |
| country | `coast` | Natural Earth | gives the shape away for islands and peninsulas |
| country | `borders` | geoBoundaries ADM1 | each country's own licence is credited |
| country | `cities` | GHSL | dots sized by population |
| city | `streets`, `rail`, `water`, `waterways` (default) | Overture or OSM | |
| city | `buildings` | Overture or OSM | figure-ground look; best with `--radius-km` 1–2 |

**Framing.**

- `--outside keep` (default, country): a square frame filled with the
  feature at full strength, neighbours included, so the silhouette is not
  visible; the border is drawn only in the answer. `dim` darkens everything
  beyond the border, `cut` removes it (easiest).
- `--padding 0.12` (country): how much surrounding area to show.
- `--budget-km 350` (city): the window grows until it holds this much
  street. `--radius-km` fixes its half-width instead.
- `--density 1.0` (country): multiplies the stroke budget.

**Look and motion.** `--style neon|ink|blueprint|ember|mono`;
`--reveal auto|network|rank|radial|sweep` (auto: rivers grow upstream from
where they leave the frame, networks grow outward along themselves, the rest
radially; `rank` draws the most important features first; `sweep` runs west
to east). Secondary features appear when the primary one reaches them.

**Text.** `--question "line one|line two"`, `--hint`, `--answer` (display
name), `--fact`. Defaults are generated; the fact is computed from the data
drawn.

**Video.** `--duration 15` or `--phases reveal,countdown,answer,outro` in
seconds (a countdown of 0 skips it), `--size 1080x1920` (any 9:16 size),
`--fps`, `--zoom`, `--crf`, `--audio music.mp3`.

**Batch file.** The same names as keys, per puzzle or under `defaults`, plus
`colors` to override a style's colours and `random` blocks for seeded picks
of countries or cities. See `examples/batch.yaml`.

## Laser engraving

`geoquiz engrave` (or `--engrave` on `render`) writes pictures for a laser
engraver, cut exactly to the target's contour: everything inside, nothing
outside. There is one picture per feature type, one for the contour, and one
with everything, all on the same canvas so they line up when engraved one
after another (for example to fill each layer with a different epoxy).

```bash
geoquiz engrave country CHE --features lakes,roads,rail --width-mm 200
geoquiz render  country CHE --features lakes,roads,rail --engrave      # video plus an engrave/ folder
geoquiz engrave city Zürich --features streets,rail,water --width-mm 150 --min-line-mm 0.25
```

| File | Content |
| --- | --- |
| `contour.svg/.png` | the outline only (for a city: the rectangular window) |
| `<feature>.svg/.png` | one feature type only, e.g. `lakes`, `roads`, `rail` |
| `all.svg/.png` | everything; in the SVG each layer is a group with its own colour, which laser software such as LightBurn maps to separate layers |
| `engrave.json` | canvas size in mm, map scale, line widths, colours, credits |

SVGs are in millimetres; PNGs are pure black and white (black = engrave) at
`--dpi` (508 = 20 pixels per mm). Options: `--width-mm` (canvas width
including `--margin-mm`), `--min-line-mm` (narrowest line: set it to the
finest groove you can engrave and fill), `--line-scale`, `--lines outline`
(default: lines become closed areas as wide as the stroke, which a laser can
fill-engrave) or `--lines centerline` (single paths for vector scoring),
`--contour-in-layers`, `--mirror` (for the back of clear material).

## Python

```python
from geoquiz import GeoQuiz
gq = GeoQuiz()                                    # data root: ./geoquiz_data

gq.countries(continent="Europe")                  # DataFrame of every country: iso3, name, continent, population
gq.cities(country="Italy", min_population=200_000)   # DataFrame of every city, largest first
gq.features("country", "Austria")                 # per feature: ready? from which source? how to get it?
gq.styles(); gq.settings()                        # style names; every setting with its default

gq.render("country", "Brazil")                                            # rivers and lakes, shape hidden
gq.render("country", "Austria", features=["roads", "rail"], style="blueprint", reveal="rank")
gq.render("city", "Paris", features=["streets", "water"], style="ink", budget_km=500,
          question="Quelle ville|est-ce ?", duration=12)
gq.render("country", "CHE", features=["lakes", "roads", "rail"], engrave=True, width_mm=200)
gq.engrave("city", "Zürich", features=["streets", "rail", "water"], width_mm=150)
gq.targets("city", "Brazil", min_population=1e6)                          # what make_all would make
gq.make_all("country", "South America", style="ember")                    # one Reel per country
gq.make_all("city", "Brazil", min_population=1e6, limit=10, video=False, engrave=True, width_mm=150)
gq.batch("geoquiz/examples/batch.yaml")
gq.ingest("hydrosheds", hydro_regions="all")
```

`render` and `engrave` download missing data by default
(`ingest_missing=False` turns that off) and return a dict with the output
`folder` and everything written to `metadata.json` / `engrave.json`.

## Data and output

Data lives under `./geoquiz_data` (override with `--data` or `$GEOQUIZ_DATA`):
`raw/` holds downloads and `manifest.json` (URL, SHA-256, size, date per
file), `store/` the normalised GeoParquet, `out/` one folder per puzzle.

Each puzzle folder holds `video.mp4` (H.264, yuv420p, AAC, moov atom first),
`cover.jpg`, `caption.txt` (with full credits), `answer.json`,
`metadata.json` (target ids, parameters and their hash, data versions,
licences, QC metrics, timings) and `data/*.parquet`, the clipped layers
behind the video, kept so they can be shared on request as ODbL asks.

## How it fits together

| Module | Role |
| --- | --- |
| `ingest.py` | Downloads raw sources once, with checksums and a manifest. Nothing is downloaded at render time. |
| `schema.py` | The common schema: `geometry, feature, rank, name, source, source_id, licence, attribution, data_version`, plus source extras as `x_*` columns. |
| `store.py` | GeoParquet store, `store/<source>/<feature>/<partition>.lod<N>.parquet`, with bbox reads and three pre-simplified levels of detail. |
| `providers/` | One adapter per source behind the `FeatureProvider` protocol; `ProviderRegistry` picks a source per feature and scale from a priority table and falls back when coverage of the region is poor. |
| `licences.py` | Licence records. Every stored row names its licence, and the on-screen credit is generated from the layers actually drawn. |
| `regions.py` | Region resolver: countries from Natural Earth, cities from GHSL urban centres; local projection; city windows sized by a density budget. |
| `scene.py` | Projection, clipping, pixel-tolerance simplification, stroke width from rank, reveal times. |
| `render/reveal.py` | The reveal engine: one crisp Skia render plus a per-pixel reveal-time map; each frame only touches pixels in transition. |
| `render/overlay.py` | Question, countdown, answer card (flag, locator globe, typed name, fact), credit line. |
| `render/encode.py` | FFmpeg pipe in the Reels format. |
| `puzzles.py` | The puzzle builder: target + features + options -> scene, text and answer. |
| `auto.py` | `--ingest-missing`: works out and downloads what a puzzle lacks. |
| `cli.py` | Commands, and the batch runner. |
| `api.py` | `GeoQuiz`: the same operations as Python calls. |
| `gui.py`, `gui.html` | The browser interface and its local server (standard library only). |
| `engrave.py` | Laser-engraving export: per-feature SVG and PNG cut to the contour. |
| `reel.py` | Timeline and assembly of one puzzle folder. |

Providers: `natural_earth`, `hydrosheds` (HydroRIVERS, HydroLAKES), `osm`
(Geofabrik PBF via pyosmium), `geoboundaries`, `ghsl`, and `overture` (roads,
rail and water by bounding box from the Overture cloud release; the fallback
for city features where no OSM extract is ingested).

## Limits to know about

- **OSM extracts from Geofabrik are untested at country size**: the host was
  unreachable where this was built. City and country OSM-derived data
  therefore came from Overture, which worked for every target tried.
- **Country roads and rail are slow for very large countries**: the Overture
  read scans the whole bounding box (Austria takes seconds, India's rail
  about a minute; the USA or Russia will take much longer).
- **Neighbours appear only where data is in the store.** Rivers, lakes, coast
  and cities are global once ingested; roads, rail and borders are per
  country, and `--ingest-missing` fetches roads and rail for the whole frame
  but borders only for the target.
- **Countries that cross the antimeridian** (Russia, Fiji) and overseas
  territories are not handled specially; far-away parts are dropped.
- **Layouts are designed for 9:16.** Other aspect ratios will render but
  text and map may collide.
- **No quality gates yet**: nothing rejects a puzzle that is too sparse,
  too generic or too easy.

## Design notes

- **Reveal order.** Rivers grow upstream from where each basin leaves the
  scene (HydroRIVERS distance-to-outlet), so trunks appear before headwaters.
  Streets grow outward from the city centre by network distance (Dijkstra on
  the street graph).
- **City centre and window.** The centre is the densest part of the street
  layer inside the GHSL urban centre; the window grows until it holds the
  street budget. Administrative outlines are not used.
- **Even density.** Country rivers are thinned per grid cell, most important
  first, so wet and dry regions get the same amount of ink; road networks
  are thinned by whole class so they stay connected.
- **Thresholds are starting values.** Reference densities for provider
  coverage, the street budget, rank cut-offs and stroke widths are estimates
  to tune once a planner produces many candidates.

## Not built yet

Quality gates, difficulty scoring, the puzzle registry and duplicate
prevention, terrain features (elevation, glaciers, mountain ranges), 3D, and
publishing.
