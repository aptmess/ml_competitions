import cv2
import albumentations as A
import logging

log = logging.getLogger(__name__)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, annotation):
        for t in self.transforms:
            img, annotation = t(img, annotation)
        return img, annotation


class Resize(object):
    def __init__(self, size):
        self.size = size
        log.info(f'Resize picture by {size}')

    def __call__(self, image, annotation):
        img = cv2.resize(image, (self.size, self.size))
        return img, annotation


class PreparedAug(object):

    def __init__(self):
        augmentation = [
            A.VerticalFlip(p=0.5),
            # A.Rotate(limit=10, p=0.5),
            # A.HueSaturationValue(hue_shift_limit=20,
            #                      sat_shift_limit=40,
            #                      val_shift_limit=50,
            #                      p=0.5),
            # A.GaussianBlur(p=0.4),
            # A.ToGray(p=0.3)
        ]
        self.augmentation = A.Compose(augmentation)

    def __call__(self, image, annotation):
        image = self.augmentation(image=image)['image']
        return image, annotation


class DefaultAugmentations(object):
    def __init__(self, config):
        self.augment = Compose([
            Resize(size=config.dataset.input_size),
            #PreparedAug()
        ])

    def __call__(self, image, annotation):
        return self.augment(image, annotation)


class ValidationAugmentations(object):
    def __init__(self, config):
        self.augment = Compose([
            Resize(size=config.dataset.input_size),
        ])

    def __call__(self, image, annotation):
        return self.augment(image, annotation)


def get_train_aug(config):
    if config.dataset.augmentations == 'default':
        train_augmentation = DefaultAugmentations(config)
    else:
        raise Exception("Unknown type of augmentation: {}".format(
            config.dataset.augmentations)
        )
    return train_augmentation


def get_val_aug(config):
    if config.dataset.augmentations_valid == 'default':
        val_augmentation = ValidationAugmentations(config)
    else:
        raise Exception("Unknown type of augmentation: {}".format(
            config.dataset.augmentations)
        )
    return val_augmentation
