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
FRUITIFY_DATASET_TSV = path.join(DATA_DIR, "fruitify_dataset.tsv")


# the models
MONO_EN_CKPT = path.join(DATA_DIR, "lightning_logs/version_0/checkpoints/mono_epoch=11_train_loss=0.04.ckpt")
CROSS_CKPT = path.join(DATA_DIR, "lightning_logs/...")

