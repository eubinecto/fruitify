"""
Paths to data, paths to saved models should be saved here
"""
from pathlib import Path
from os import path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(PROJECT_ROOT, "data")
SAVED_DIR = path.join(DATA_DIR, "saved")
