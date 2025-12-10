"""
Centralized file and directory paths for the project.
All paths are absolute to avoid ambiguity when scripts are run from different working directories.
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Core assets
ROM_PATH = PROJECT_ROOT / "pokemon-red.gb"
INIT_STATE_PATH = PROJECT_ROOT / "init.state"
AGENT_ENABLED_PATH = SRC_DIR / "agent_enabled.txt"

# Data files
EVENTS_PATH = SRC_DIR / "events.json"
MAP_DATA_PATH = SRC_DIR / "map_data.json"

# Output directories
RUNS_DIR = SRC_DIR / "runs"
