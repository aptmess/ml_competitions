import torch
import logging
import numpy as np
import torch.utils.data

from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import f1_score

import sys

sys.path.append('.')

from .average_meter import AverageMeter

log = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = - log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = - log_prob.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_loss(config, device='cuda'):
    if config.train.label_smoothing:
        criterion = LabelSmoothingCrossEntropy(config.train.eps).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    criterion_val = torch.nn.CrossEntropyLoss().to(device)

    return criterion, criterion_val


def get_optimizer(config, opt, net):
    lr = config.train.learning_rate
    log.info(lr)

    log.info(f"Opt: {opt}")

    if opt == 'SGD':
        optimizer = torch.optim.SGD(
            filter(
                lambda p: p.requires_grad,
                net.parameters()
            ),
            lr=lr,
            momentum=config.train.momentum
        )
    elif opt == 'Adam':
        optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                net.parameters()
            ),
            lr=0.001
        )
    else:
        raise Exception("Unknown type of optimizer: {}".format(
            opt)
        )
    return optimizer


def get_scheduler(config, optimizer):
    if config.train.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.n_epoch
        )
    elif config.train.lr_schedule == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=5
        )
    else:
        raise Exception("Unknown type of lr schedule: {}".format(
            config.train.lr_schedule)
        )
    return scheduler


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          config,
          epoch: int) -> None:
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch: epoch number
    :return: None
    """
    model.train()

    loss_stat = AverageMeter('Loss')
    f1_stat = AverageMeter('F1.')

    train_iter = tqdm(
        train_loader,
        desc='Train',
        dynamic_ncols=True,
        position=1
    )

    for step, (x, y) in enumerate(train_iter):
        out = model(x.cuda().to(memory_format=torch.contiguous_format))
        loss = criterion(out, y.cuda())
        num_of_samples = x.shape[0]

        loss_stat.update(loss.detach().cpu().item(), num_of_samples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores = torch.softmax(out, dim=1).detach().cpu().numpy()
        predict = np.argmax(scores, axis=1)
        gt = y.detach().cpu().numpy()
        f1 = f1_score(gt, predict, average='weighted')
        f1_stat.update(f1, num_of_samples)

        if step % config.train.freq_vis == 0 and not step == 0:
            f1_val, f1_avg = f1_stat()
            loss_val, loss_avg = loss_stat()
            print('Epoch: {}; step: {}; loss: {:.4f}; f1: {:.4f}'.format(
                epoch,
                step,
                loss_avg,
                f1_avg)
            )

    f1_val, f1_avg = f1_stat()
    loss_val, loss_avg = loss_stat()
    print(
        'Train process of epoch'
        ': {} is done; \n loss: {:.4f}; f1: {:.4f}'.format(
            epoch,
            loss_avg,
            f1_avg
        )
    )


def validation(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        epoch: int
) -> float:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch: epoch number
    :return: None`
     """
    loss_stat = AverageMeter('Loss')
    f1_stat = AverageMeter('F1.')

    with torch.no_grad():
        model.eval()
        val_iter = tqdm(val_loader, desc='Val', dynamic_ncols=True, position=2)

        for step, (x, y) in enumerate(val_iter):
            out = model(x.cuda().to(memory_format=torch.contiguous_format))
            loss = criterion(out, y.cuda())
            num_of_samples = x.shape[0]

            loss_stat.update(loss.detach().cpu().item(), num_of_samples)

            scores = torch.softmax(out, dim=1).detach().cpu().numpy()
            predict = np.argmax(scores, axis=1)
            gt = y.detach().cpu().numpy()

            f1 = f1_score(gt, predict, average='weighted')
            f1_stat.update(f1, num_of_samples)

        f1_val, f1_avg = f1_stat()
        loss_val, loss_avg = loss_stat()
        print(
            'Validation of epoch'
            ': {} is done; \n loss: {:.4f}; f1: {:.4f}'.format(
                epoch,
                loss_avg,
                f1_avg
            )
        )
        return f1_avg
