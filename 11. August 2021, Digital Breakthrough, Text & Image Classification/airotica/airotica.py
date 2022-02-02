# IMPORT
import cv2
import random
import os
import yaml
import torch
import logging
import numpy as np
import pandas as pd
import albumentations as A

# FROM
from tqdm import tqdm
from torchvision import transforms as tfs
from pathlib import Path
from airotica.augmentations import ValidationAugmentations

from airotica.utils import (
    load_resnet,
    save_results,
    convert_dict_to_tuple
)
from definitions import ROOT_DIR


log = logging.getLogger(__name__)


def detectron(config_name,
              save_path: str,
              model_path: str,
              model_type: str,
              number_of_images=10,
              experiment=False,
              use_description=True,
              save_result=True,
              return_result=True):

    with open(ROOT_DIR / config_name) as f:
        data = yaml.safe_load(f)
    config = convert_dict_to_tuple(dictionary=data)
    # device = torch.device(device_name)
    # print(f'device: {device_name}')
    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_resnet(
        path=model_path,
        model_type=model_type,
        num_classes=config.dataset.num_of_classes,
        device=device_name
    )

    softmax_func = torch.nn.Softmax(dim=1)
    validation_augmentation = ValidationAugmentations(config=config)
    preprocess = tfs.Compose(
        [
            tfs.ToTensor(),
            tfs.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    test_df = pd.read_csv(config.path.input_path)
    test_image_path = Path(config.path.test_images_path)
    save_p = Path(save_path)
    len_ = len(test_df)
    if not experiment:
        scores = np.zeros((len_, 1), dtype=np.float32)
        for idx, row in tqdm(test_df.iterrows(), total=len_):
            if use_description:
                if not pd.isnull(row['description']):
                    scores[idx] = -1
                    continue

            guid = row['guid']
            path = str(test_image_path / f'{guid}.jpg')
            img = cv2.imread(path,
                             cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if img is None:
                print(f'None img {path}')
                scores[idx] = -1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            crop, _ = validation_augmentation(
                image=img,
                annotation=None
            )
            crop = preprocess(crop).unsqueeze(0)
            crop = crop.to(device_name)
            out = model(crop)
            out = softmax_func(out).squeeze().detach().cpu().numpy()
            scores[idx] = np.argmax(out)

            if idx % 50 == 0 and idx > 0 and save_result:
                save_results(
                    scores=scores,
                    frame_paths=test_df.guid.values,
                    save_path=save_p
                )
        result_df = save_results(
            scores=scores,
            frame_paths=test_df.guid.values,
            save_path=save_p,
            save_result=save_result,
            return_result=return_result
        )
        return result_df
    else:
        scores = np.zeros((len_, number_of_images), dtype=np.float32)
        save_p = Path(save_path)
        augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=40,
                                 val_shift_limit=50,
                                 p=0.5),
            A.GaussianBlur(p=0.4),
            A.ToGray(p=0.3)
        ])
        for idx, row in tqdm(test_df.iterrows(), total=len_):
            if use_description:
                if not pd.isnull(row['description']):
                    scores[idx] = np.tile(-1, number_of_images)
                    continue

            guid = row['guid']
            path = str(test_image_path / f'{guid}.jpg')
            img = cv2.imread(path,
                             cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if img is None:
                print(f'None img {path}')
                continue

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for i in range(number_of_images):
                im = augmentation(image=img)['image']
                crop, _ = validation_augmentation(
                    image=im,
                    annotation=None
                )
                crop = preprocess(crop).unsqueeze(0)
                crop = crop.to(device_name)
                out = model(crop)
                out = softmax_func(out).squeeze().detach().cpu().numpy()
                scores[idx, i] = np.argmax(out)

            if idx % 50 == 0 and idx > 0:
                save_results(
                    scores=scores,
                    frame_paths=test_df.guid.values,
                    save_path=save_p
                )

        result_df = save_results(
            scores=scores,
            frame_paths=test_df.guid.values,
            save_path=save_p,
            return_result=True
        )
        return result_df
