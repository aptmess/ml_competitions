import torch
import torch.utils.data
import logging

import sys

sys.path.append('.')

from .utils import read_image
from .dataset import Dataset

log = logging.getLogger(__name__)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_data_loaders(train_data,
                     valid_data,
                     config,
                     train_img_path,
                     valid_img_path,
                     read_image=read_image):
    log.info("Preparing train reader...")
    train_dataset = Dataset(
        data=train_data,
        config=config,
        img_path=train_img_path,
        is_train=True,
        read_image=read_image
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    log.info("Train loader.. Done.")

    log.info("Preparing valid reader...")
    val_dataset = Dataset(
        data=valid_data,
        config=config,
        img_path=valid_img_path,
        is_train=False,
        read_image=read_image
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    log.info("Valid loader.. Done.")
    return train_loader, valid_loader
