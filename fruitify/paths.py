"""
Paths to data, paths to saved models should be saved here
"""
from pathlib import Path
from os import path

# the directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(PROJECT_ROOT, "data")
SAVED_DIR = path.join(DATA_DIR, "saved")


# the files.
FRUIT2DEFS_TSV = path.join(DATA_DIR, "fruit2def.tsv")


# the models
MONO_CKPT = path.join(DATA_DIR, "lightning_logs/version_12/checkpoints/mono_epoch=08_train_loss=1.59.ckpt")


