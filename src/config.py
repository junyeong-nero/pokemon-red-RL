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
MAP_MODULE_PATH = RESOURCES_DIR / "processed_map"

# Custom Data files
CHARMAP_PATH = RESOURCES_DIR / "charmap.json"
ITEM_NAMES_PATH = RESOURCES_DIR / "item_names.json"
MAP_NAMES_PATH = RESOURCES_DIR / "map_names.json"
SPECIES_NAMES_PATH = RESOURCES_DIR / "species_names.json"
TYPE_NAMES_PATH = RESOURCES_DIR / "type_names.json"
MOVE_NAMES_PATH = RESOURCES_DIR / "move_names.json"

ASM_DIR = RESOURCES_DIR / "asm"

# Data files
EVENTS_PATH = RESOURCES_DIR / "events.json"
MAP_DATA_PATH = RESOURCES_DIR / "map_data.json"

# Output directories
RUNS_DIR = SRC_DIR / "runs"

# Agent toggle for interactive play; set to True to let the model act.
AGENT_ENABLED = True
