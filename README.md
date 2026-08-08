# swiss_rail_rl

A reinforcement-learning agent discovers a near-optimal rail network
connecting ~20 Swiss cities, then the video reveals how closely its learned
network tracks the real Swiss rail network.

The agent connects fixed city points using edges whose cost is horizontal
length plus a terrain penalty derived from real elevation data (going over
the Alps is expensive, never forbidden). It may also add extra **junction**
(Steiner) points to branch efficiently. Because a length-only optimum
produces classic ~120° Steiner-tree geometry and a terrain-aware optimum
bends around mountains, the learned network should visibly resemble the real
one — the video's payoff is overlaying the two.

Every tunable parameter lives in `config.yaml`. On-screen video text lives in
`captions.yaml`. Nothing else needs to be edited to change cities, cost
weights, training budget, or render settings.

## Install

```bash
pip install -e .
# optional, for the polished manim reveal instead of the matplotlib one:
pip install -e ".[manim]"
```

Requires `ffmpeg` support for MP4 encoding, provided by the bundled
`imageio[ffmpeg]` dependency — no system ffmpeg install needed.

## Run

Each pipeline stage (§3 of the build spec) is a CLI subcommand, independently
re-runnable and cheap to re-run from cache:

```bash
swiss-rail-rl fetch            # download/cache cities, elevation, real rail network
swiss-rail-rl fit              # fit cost weights against the real network -> out/fitted_weights.yaml
swiss-rail-rl train            # train the RL agent, logging frames to out/frames/
swiss-rail-rl render           # render out/frames/ + the reveal to an MP4
swiss-rail-rl render --landscape  # 16:9 variant

swiss-rail-rl all              # fetch -> fit -> train -> render, in order

swiss-rail-rl baseline         # fast sanity check: MST + Steiner heuristic costs only, no RL/render
```

Pass `--config path/to/config.yaml` to any subcommand to use a different
config file (e.g. a smaller point set / shorter training run for quick
iteration). `train` and `all` accept `--no-resume` to ignore an existing
checkpoint and start fresh.

All downloaded/generated data is cached under `data.cache_dir` (default
`./data_cache/`); training frames, logs, models, fitted weights, and the
final video go under `./out/`.

## How it fits together

```
config.yaml + captions.yaml
        |
        v
swiss_rail_rl/data/{cities,elevation,rail,water}.py   -- fetch/cache raw data
        |
        v
swiss_rail_rl/cost.py (CostOracle)                    -- the only place edge/network cost is computed
        |
        +--> swiss_rail_rl/baselines.py   -- MST + greedy Steiner heuristic (fast, used as a sanity
        |                                     check and as fit.py's cheap inner-loop solver)
        |
        +--> swiss_rail_rl/fit.py         -- fits cost.slope_weight (etc.) so the baseline-optimal
        |                                     network matches the real one -> out/fitted_weights.yaml
        |
        +--> swiss_rail_rl/env.py         -- Gymnasium env wrapping CostOracle for sequential
        |         |                          network construction
        |         v
        +--> swiss_rail_rl/train.py       -- SB3 PPO on the env, snapshotting the network every
        |                                     checkpoint into out/frames/*.json + training_log.csv
        |
        v
swiss_rail_rl/render.py                   -- animates the frame log, then fades in the real network
        |
        v
out/swiss_rail_rl.mp4
```

## Baseline sanity check

Before spending RL compute, always look at `swiss-rail-rl baseline` (or the
`baselines.json` that `train` writes automatically): it prints the MST and
Steiner-heuristic costs for the current config. The RL agent should approach
or beat the Steiner cost after enough training — a persistent large gap
means the reward shaping or action space needs fixing, not more compute
(§6, §13 of the build spec).

## Data sources and offline fallbacks

- **Cities**: taken directly from `config.yaml` (WGS84), projected to
  `region.crs_projected` (Swiss LV95 by default). No download needed.
- **Elevation**: tries the configured source, in practice fetching public
  Terrarium/SRTM-derived tiles and warping them into the metric CRS. If
  elevation data can't be reached at all, falls back to a procedurally
  generated terrain shaped like Switzerland's coarse geography (flat
  Mittelland band, Alpine wall to the south, Jura ridge to the north-west) so
  the rest of the pipeline still runs — a warning is logged either way.
- **Real rail network**: tries swissTLM3D (opendata.swiss) or the OSM
  Switzerland-Railways export (HDX), per `data.rail_source`. If neither is
  reachable, falls back to an *approximate offline topology sketch* — straight
  segments between the actual configured city coordinates, following the real
  SBB trunk-line pattern (Gotthard, Lötschberg/Simplon, the Mittelland spine,
  etc). This is clearly logged as non-authoritative; re-run `fetch` once
  network access to the real source is available and the cache will refresh.
- **Water**: disabled by default (`data.water_enabled: false`). If enabled
  without a real hydrography source reachable, falls back to coarse circular
  stand-ins for the largest Swiss lakes.

All fallbacks emit a `UserWarning` so it's always clear which run used real
data versus a stand-in.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Unit tests cover the cost function, the MST/Steiner baselines, and the
Gymnasium environment — all synthetic/offline, no network or GPU required.
