# geoquiz

Automated geo-quiz Reels: "which country is this, from its rivers?", "which
city is this, from its streets?". An offline Python pipeline on pre-downloaded
open data, rendered with Skia + NumPy/OpenCV and encoded with FFmpeg.

This package covers **phase 0** (two finished test Reels) and **phase 1**
(provider adapters, GeoParquet store, region resolver, per-layer licence
metadata) of the research report's roadmap. It is independent of
`swiss_rail_rl` and shares only the repository.

## Install

```bash
pip install -e ".[geoquiz,dev]"     # from the repository root
```

Needs `ffmpeg` on the PATH (or `pip install imageio-ffmpeg`), and on a
headless Linux box `libegl1` for skia-python. Text uses the Inter and Inter
Display fonts when installed and falls back to the system default otherwise.

## Run

```bash
geoquiz ingest                                         # Natural Earth, HydroSHEDS (Europe), GHSL, flags
geoquiz ingest --sources geoboundaries --country CHE   # prints each boundary's own licence
geoquiz ingest --sources osm --osm-extract europe/switzerland    # Geofabrik extract
geoquiz ingest --sources overture --city Zürich        # alternative to an OSM extract, one city at a time

geoquiz inventory                 # what the store holds, and the licences in play
geoquiz resolve city Zürich       # region, projection, and which provider covers each feature
geoquiz render country CHE        # one Reel
geoquiz render city Zürich --budget-km 350
geoquiz spike                     # phase 0: both of the above
pytest tests/geoquiz              # offline, synthetic data
```

Data lives under `./geoquiz_data` (override with `--data` or `$GEOQUIZ_DATA`):
`raw/` holds downloads and `manifest.json` (URL, SHA-256, size, date per
file), `store/` the normalised GeoParquet, `out/` one folder per puzzle. The
default ingest downloads about 1.2 GB, most of it HydroLAKES.

Each puzzle folder holds `video.mp4` (1080 × 1920, 30 fps, H.264, yuv420p,
AAC, moov atom first), `cover.jpg`, `caption.txt` (with full credits),
`answer.json`, `metadata.json` (target ids, parameters and their hash, data
versions, licences, QC metrics, timings) and `data/*.parquet`, the clipped
layers behind the video, kept so they can be shared on request as ODbL asks.

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
| `puzzles.py` | The two puzzle builders: `country_rivers`, `city_streets`. |
| `reel.py` | Timeline and assembly of one puzzle folder. |

Providers: `natural_earth`, `hydrosheds` (HydroRIVERS, HydroLAKES), `osm`
(Geofabrik PBF via pyosmium), `geoboundaries`, `ghsl`, and `overture` (roads,
rail and water by bounding box from the Overture cloud release; the fallback
for city features where no OSM extract is ingested).

## Design notes

- **Reveal order.** Rivers grow upstream from where each basin leaves the
  scene (HydroRIVERS distance-to-outlet), so trunks appear before headwaters.
  Streets grow outward from the city centre by network distance (Dijkstra on
  the street graph).
- **City centre and window.** The centre is the densest part of the street
  layer inside the GHSL urban centre; the window grows until it holds the
  street budget. Administrative outlines are not used.
- **Difficulty knob (country mode).** `--outside dim` keeps rivers beyond the
  border at 25 % brightness so the silhouette stays hidden; `--outside cut`
  removes them.
- **Thresholds are starting values.** Reference densities for provider
  coverage, the street budget, rank cut-offs and stroke widths are estimates
  to tune once a planner produces many candidates.

## Not built yet

Planner and quality gates, difficulty scoring, batch config and registry
table, more features and styles, music, and publishing (roadmap phases 2–5).
