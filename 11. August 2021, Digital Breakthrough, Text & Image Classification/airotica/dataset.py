import logging
import cv2
import torch
import torch.utils.data
from torchvision import transforms as tfs
from airotica.augmentations import (
    get_train_aug,
    get_val_aug
)
from airotica.utils import read_image

log = logging.getLogger(__name__)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class GestureDataset(object):
    def __init__(self,
                 data,
                 config,
                 is_train,
                 img_path,
                 download_img_path='',
                 use_data=False):
        self.data = data
        if is_train:
            self.transforms = get_train_aug(config)
        else:
            self.transforms = get_val_aug(config)
        self.img_path = img_path
        self.use_add_data = use_data
        self.download_img_path = download_img_path
        self.preprocess = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        self.dataset_length = len(self.data)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        cv2.setNumThreads(6)
        sample = self.data.iloc[idx]
        image_name = sample['guid']
        if self.use_add_data:
            if sample['add'] == 0:
                path = lambda form: str(self.img_path / f'{image_name}.{form}')
            else:
                path = lambda form: str(
                    self.download_img_path / f'{image_name}.{form}'
                )
        else:
            path = lambda form: str(self.img_path / f'{image_name}.{form}')
        m = ['jpg', 'jpeg', 'png', 'psd', 'tiff',
             'bmp', 'gif', 'eps', 'pict', 'pdf',
             'pcx', 'ico', 'cdr', 'ai', 'raw', 'svg']
        count = 0
        for i in m:
            try:
                image = read_image(path(i))
                break
            except Exception as ex:
                count += 1
        if count == len(m):
            return None
        crop, _ = self.transforms(image, None)
        crop = self.preprocess(crop)
        return crop, sample['label'], image_name


def get_data_loaders(train_data,
                     valid_data,
                     config,
                     img_path,
                     download_img_path='',
                     use_data=False):
    log.info("Preparing train reader...")
    train_dataset = GestureDataset(
        data=train_data,
        config=config,
        img_path=img_path,
        is_train=True,
        download_img_path=download_img_path,
        use_data=use_data

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
    log.info("Done.")
    log.info("here")

    log.info("Preparing valid reader...")
    val_dataset = GestureDataset(
        data=valid_data,
        config=config,
        img_path=img_path,
        is_train=False,
        download_img_path=download_img_path,
        use_data=use_data
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
    log.info("Done.")
    return train_loader, valid_loader
