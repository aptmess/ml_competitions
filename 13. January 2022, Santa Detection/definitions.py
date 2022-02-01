from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / 'data'

TRAIN_IMAGES = DATA_DIR / 'train' / 'images'

TRAIN_PHOTOS_DIR = DATA_DIR / 'train' / 'train'

TRAIN_LABELS_DIR = DATA_DIR / 'train.csv'

TEST_DIR = DATA_DIR / 'test'

OUT_DIR = DATA_DIR / 'out'

WEIGHTS_DIR = DATA_DIR / 'weight'

CONFIG_PATH = ROOT_DIR / 'config.yml'

LOG_CONFIG = ROOT_DIR / 'log_config.yml'
