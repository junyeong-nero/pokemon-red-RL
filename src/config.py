"""
Centralized file and directory paths for the project.
All paths are absolute to avoid ambiguity when scripts are run from different working directories.
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Core assets
ROM_PATH = RESOURCES_DIR / "pokemon-red.gb"
INIT_STATE_PATH = RESOURCES_DIR / "init.state"

# Data files
EVENTS_PATH = RESOURCES_DIR / "events.json"
MAP_DATA_PATH = RESOURCES_DIR / "map_data.json"

# Output directories
RUNS_DIR = SRC_DIR / "runs"

# Agent toggle for interactive play; set to True to let the model act.
AGENT_ENABLED = True
