# Repository Guidelines

## Project Structure & Modules
- `src/`: primary code (gym envs, training scripts, config, monitor). Key entries: `red_gym_env.py`, `train_agent.py`, `inference.py`, `config.py`, `monitor.py`.
- `resources/`: JSON assets (`events.json`, `map_data.json`).
- `runs/` (under `src/`): checkpoints and training outputs; inference reuses this folder.
- `outputs/`: mp4 recordings from `inference.py` via `VideoMonitor`.
- `scripts/`: wrappers (`train.sh`, `inference.sh`).
- `visualization/` and notebooks: analysis and visuals.

## Build, Test, and Development Commands
- Install deps (v2): `cd src && pip install -r requirements.txt` (macOS: `macos_requirements.txt`).
- Run pretrained interactively: `scripts/inference.sh` (loads latest checkpoint in `src/runs` and writes video to `outputs/`).
- Train v2: `scripts/train.sh` or `uv run src/train_agent.py`.
- Verify ROM hash (from repo root): `shasum resources/pokemon-red.gb`.

## Coding Style & Naming
- Python, 4-space indentation; follow PEP 8 where unspecified.
- Keep paths centralized via `src/config.py`; avoid hardcoding asset/ROM/checkpoint paths.
- Prefer pathlib over string paths; use explicit names for checkpoints (e.g., `poke_<steps>.zip`).

## Testing Guidelines
- No formal test suite yet. When adding tests, place under `tests/` or alongside modules; use `pytest`. Keep runtime short and avoid GPU-heavy defaults.

## Commit & PR Guidelines
- Commits: concise imperative summary (e.g., “add video monitor for inference”); group related changes.
- PRs: describe intent, key changes, and testing performed; note checkpoint or asset expectations. Include screenshots/gifs for UI/visual output when relevant (e.g., map/recording previews).

## Agent & Resource Notes
- ROM and init state paths are defined in `config.py` (default under `resources/`); avoid checking these binaries into version control.
- Checkpoints are read from `src/runs`; keep large artifacts out of commits.
- Recording uses `outputs/`; ensure the directory is writable during inference.
