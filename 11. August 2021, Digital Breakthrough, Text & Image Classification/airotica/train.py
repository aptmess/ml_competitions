import torch
import torch.utils.data
import numpy as np
import logging
from tqdm import tqdm
from airotica.averagemeter import AverageMeter
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          config,
          epoch) -> None:
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch: epoch number (int)
    :return: None
    """
    model.train()

    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')
    roc_auc = AverageMeter('F1-Score')

    train_iter = tqdm(train_loader,
                      desc='Train',
                      dynamic_ncols=False,
                      position=1)

    for step, (x, y, guid) in enumerate(train_iter):
        # if x is None:
        #     log.info(f'train: failed reading {guid}')
        #     continue
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

        acc = np.mean(gt == predict)
        acc_stat.update(acc, num_of_samples)
        roc_auc.update(
            f1_score(gt, predict, average='weighted'),
            num_of_samples
        )
        if step % config.train.freq_vis == 0 and not step == 0:
            acc_val, acc_avg = acc_stat()
            loss_val, loss_avg = loss_stat()
            roc_val, roc_avg = roc_auc()
            print("""
            Epoch: {}; step: {}; loss: {:.4f}; acc: {:.2f}, f1: {:.8f}
            """.format(epoch, step, loss_avg, acc_avg, roc_avg))

    acc_val, acc_avg = acc_stat()
    loss_val, loss_avg = loss_stat()
    roc_val, roc_avg = roc_auc()
    print("""
    Train process of epoch: {} is done;
    loss: {:.4f}; acc: {:.2f}, f1: {:.8f}
    """.format(epoch, loss_avg, acc_avg, roc_avg))


def validation(model: torch.nn.Module,
               val_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               epoch) -> None:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch: epoch number (int)
    :return: None`
     """
    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')
    roc_auc = AverageMeter('F1-Score')

    with torch.no_grad():
        model.eval()
        val_iter = tqdm(val_loader,
                        desc='Val',
                        dynamic_ncols=False,
                        position=2)

        for step, (x, y, guid) in enumerate(val_iter):
            # if x is None:
            #     log.info(f'validation: failed reading {guid}')
            #     continue
            out = model(x.cuda().to(memory_format=torch.contiguous_format))
            loss = criterion(out, y.cuda())
            num_of_samples = x.shape[0]

            loss_stat.update(loss.detach().cpu().item(), num_of_samples)

            scores = torch.softmax(out, dim=1).detach().cpu().numpy()
            predict = np.argmax(scores, axis=1)
            gt = y.detach().cpu().numpy()

            acc = np.mean(gt == predict)
            acc_stat.update(acc, num_of_samples)
            roc_auc.update(
                f1_score(gt, predict, average='weighted'),
                num_of_samples
            )

        acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        roc_val, roc_avg = roc_auc()
        print("""
        Validation of epoch: {} is done; 
        loss: {:.4f}; acc: {:.2f}, f1: {:.8f}
        """.format(epoch, loss_avg, acc_avg, roc_avg))
        return roc_avg
