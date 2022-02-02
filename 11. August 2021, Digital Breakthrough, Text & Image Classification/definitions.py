from pathlib import Path

# ./
ROOT_DIR = Path(__file__).resolve().parent

# ./data
DATA_PATH = ROOT_DIR / 'data'

# ./data/images
TRAIN_IMAGES = DATA_PATH / 'images'

# ./data/downloaded_images
DOWNLOADED_TRAIN_IMAGES = DATA_PATH / 'downloaded_images'

# ./data/more_images

MORE_IMAGES = DATA_PATH / 'more_images'

# ./data/description
DESCRIPTION_PATH = DATA_PATH / 'description'

# ./data/img_data
IMG_PATH = DATA_PATH / 'img_data'

# ./data/models
MODELS_PATH = ROOT_DIR / 'models'

# ./data/models/lm_models
LM_MODELS_PATH = MODELS_PATH / 'lm_models'

# ./data/models/model

CLASSIFICATION_MODELS_PATH = MODELS_PATH / 'model'

# ./data/models/tokenizers

TOKENIZERS_PATH = MODELS_PATH / 'tokenizers'

# ./data/src

SRC_PATH = ROOT_DIR / 'src'

# ./data/src/configs

CONFIGS_PATH = SRC_PATH / 'configs'

# ./data/src/prediction_configs

PREDICTION_CONFIGS_PATH = SRC_PATH / 'prediction_configs'

# ./sub

SUB_PATH = ROOT_DIR / 'sub'
