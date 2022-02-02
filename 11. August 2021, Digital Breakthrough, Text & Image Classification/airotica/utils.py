import os
import cv2
import torch
import logging
import pandas as pd
from collections import namedtuple, OrderedDict
from torchvision import models

log = logging.getLogger(__name__)


def read_image(image_path):
    img = cv2.imread(image_path,
                     cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def load_resnet(path,
                model_type,
                num_classes,
                device='cuda'):
    if model_type == 'resnet34':
        model = models.resnet34(pretrained=False)

    elif model_type == 'resnext101_32x8d':
        model = models.resnext101_32x8d(pretrained=False)

    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=False)

    else:
        raise Exception("Unknown model type: {}".format(model_type))

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(
        torch.load(path, map_location='cpu')["state_dict"]
    )

    model.to(device)

    model.eval()

    return model


def save_results(scores,
                 frame_paths,
                 save_path,
                 save_result=True,
                 return_result=False):
    d = {**{f'sign_{i}': scores[:, i] for i in range(scores.shape[1])},
         **{'filename': frame_paths}}
    result_df = pd.DataFrame(d)
    if save_result:
        result_df.to_csv(save_path, index=False)
    if return_result:
        return result_df


def save_checkpoint(model, optimizer, scheduler, epoch, outdir, roc_auc):
    """Saves checkpoint to disk"""
    filename = "model_{:04d}_{:04f}.pth".format(epoch, roc_auc)
    directory = outdir
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict([
        ('state_dict', weights),
        ('optimizer', optimizer.state_dict()),
        ('scheduler', scheduler.state_dict()),
        ('epoch', epoch),
    ])

    torch.save(state, filename)
    return filename
