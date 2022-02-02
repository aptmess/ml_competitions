import logging
import torch
from torch import nn
from torch.nn import functional as F

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
