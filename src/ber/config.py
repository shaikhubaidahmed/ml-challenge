"""Configuration loading and runtime helpers."""
import ctypes
import os
from pathlib import Path

import yaml

# Relative paths in configs resolve against the project folder (business_entity_resolution/).
REPO_ROOT = Path(__file__).resolve().parents[2]
# configs/ lives next to src/ in the dev repo and inside src/ in the submission package.
CONFIG_DIR = next((d for d in (REPO_ROOT / "configs", REPO_ROOT / "src" / "configs") if d.exists()), REPO_ROOT / "configs")


def load_config(path=None):
    """Load a YAML config and resolve relative paths against the repo root."""
    path = Path(path) if path else CONFIG_DIR / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else (REPO_ROOT / p).resolve()
    Path(cfg["paths"]["work_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def preload_libomp():
    """LightGBM wheels on macOS need libomp. Load it from LIBOMP_PATH or common locations if present."""
    candidates = [os.environ.get("LIBOMP_PATH"), "/opt/homebrew/opt/libomp/lib/libomp.dylib",
                  "/opt/anaconda3/lib/libomp.dylib", "/usr/local/opt/libomp/lib/libomp.dylib"]
    for c in candidates:
        if c and os.path.exists(c):
            ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
            return c
    return None
