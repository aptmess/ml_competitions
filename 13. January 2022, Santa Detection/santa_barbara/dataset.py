import logging
import cv2
import pandas as pd
from pathlib import Path

import sys

sys.path.append('.')

from .augmentate import get_train_aug, get_val_aug
from .constants import PREPROCESS

log = logging.getLogger(__name__)


class Dataset(object):
    def __init__(
            self,
            data: pd.DataFrame,
            config: dict,
            img_path: Path,
            is_train: bool,
            read_image: callable
    ):
        self.data = data
        if is_train:
            self.transforms = get_train_aug(config)
        else:
            self.transforms = get_val_aug(config)
        self.img_path = img_path
        self.preprocess = PREPROCESS
        self.dataset_length = len(self.data)
        self.read_image = read_image

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        cv2.setNumThreads(6)
        sample = self.data.iloc[idx]
        image_name = sample['image_name']
        path = str(self.img_path / image_name)
        image = self.read_image(path)
        clss = sample['class_id']
        crop, _ = self.transforms(image, clss)
        crop = self.preprocess(crop)
        return crop, clss
