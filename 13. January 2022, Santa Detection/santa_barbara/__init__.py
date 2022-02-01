import yaml
from logging import config

import sys

sys.path.append('.')

from .augmentate import *
from .average_meter import *
from .constants import *
from .data_loader import *
from .dataset import *
from .train import *
from .utils import *


with open(Path(__file__).parents[1] / 'log_config.yml') as f:
    log_config = yaml.load(f, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
