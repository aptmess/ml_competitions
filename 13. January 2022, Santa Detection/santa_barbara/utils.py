import os
import cv2
import torch
import logging
from torch import nn
from torchvision import models
from collections import (
    namedtuple,
    OrderedDict
)

log = logging.getLogger(__name__)


def read_image(image_path):
    img = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if img is None:
        raise ValueError('Failed to read {}'.format(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def set_requires_grad(model, value=False):
    for param in model.parameters():
        param.requires_grad = value


def load_model(model_type,
               device='cuda', requires_grad=False):
    log.info(f'Loading model {model_type}')
    if model_type == 'resnet34':
        model = models.resnet34(pretrained=True)
        set_requires_grad(model, requires_grad)
        model.fc = nn.Linear(
            model.fc.in_features,
            3
        )

    elif model_type == 'resnext101_32x8d':
        model = models.resnext101_32x8d(pretrained=True)
        set_requires_grad(model, requires_grad)
        model.fc = nn.Linear(
            model.fc.in_features,
            3
        )

    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=True)
        set_requires_grad(model, requires_grad)
        model.fc = nn.Linear(
            model.fc.in_features,
            3
        )

    elif model_type == 'efficientnet_b7':
        model = models.efficientnet_b7(pretrained=True)
        # set_requires_grad(model, requires_grad)
        model.classifier = nn.Linear(
            model.classifier[1].in_features,
            3
        )
    elif model_type == 'efficientnet_b6':
        model = models.efficientnet_b6(pretrained=True)
        # set_requires_grad(model, requires_grad)
        model.classifier = nn.Linear(
            model.classifier[1].in_features,
            3
        )
        #     ),
        # model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(
        #         model.classifier[1].in_features,
        #         512
        #     ),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 256),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(
        #         256, 
        #         config.dataset.num_of_classes
        #         )
        # )
        # model = EfficientNetB7()
        # model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(2560, 512),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 256),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(
        #         in_features=256,
        #         out_features=config.dataset.num_of_classes,
        #         bias=True
        #     )
        # )

    else:
        raise Exception('model type is not supported:',
                        model_type)
    model.to(device)
    return model


def load_resnet(path,
                model_type,
                num_classes,
                device='cuda'):
    log.info(f'Loading model {model_type} with {num_classes} '
             f'classes on {device} device')
    if model_type == 'resnet34':
        model = models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'resnext101_32x8d':
        model = models.resnext101_32x8d(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'efficientnet_b7':
        model = models.efficientnet_b7(pretrained=False)
        model.classifier = nn.Linear(
            2560,
            num_classes
        )
    elif model_type == 'efficientnet_b6':
        model = models.efficientnet_b6(pretrained=False)
        model.classifier = nn.Linear(
            2304,
            num_classes
        )
        # model.classifier = nn.Sequential(
        #     nn.Linear(2560, 512),
        #     nn.Dropout(p=0.5, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.Linear(
        #         in_features=256,
        #         out_features=num_classes,
        #         bias=True
        #     )
        # )
    else:
        raise Exception("Unknown model type: {}".format(model_type))
    model.load_state_dict(
        torch.load(path, map_location='cpu')["state_dict"]
    )
    model.to(device)
    model.eval()
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, filename, f1_avg):
    """Saves checkpoint to disk"""
    # filename = "model_{:04d}_{:04f}.pth".format(epoch, roc_auc)
    weights = model.state_dict()
    state = OrderedDict([
        ('state_dict', weights),
        ('optimizer', optimizer.state_dict()),
        ('scheduler', scheduler.state_dict()),
        ('epoch', epoch),
    ])

    torch.save(state, filename)
    log.info(f'Saved model with better f1 - {f1_avg}')
    return filename

