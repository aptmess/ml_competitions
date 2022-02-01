import os
import yaml
import torch
import random
import logging
import pandas as pd
import numpy as np
import torch.backends.cudnn
import santa_barbara as sb
from sklearn.model_selection import train_test_split, StratifiedKFold

from definitions import (
    CONFIG_PATH,
    TRAIN_LABELS_DIR,
    TRAIN_PHOTOS_DIR,
    ROOT_DIR,
    DATA_DIR,
    WEIGHTS_DIR
)
from tqdm import tqdm

log = logging.getLogger(__name__)


def train_model_step_1():
    log.info('TRAINING MODEL FRO STEP 1')
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    config = sb.convert_dict_to_tuple(dictionary=data)

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device_name)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id

    log.info(f'device: {device_name}')

    log.info("Loading model...")

    skf = StratifiedKFold(n_splits=4)

    train_data = pd.read_csv(TRAIN_LABELS_DIR, sep='\t')
    #  train_data['class_id'] = train_data['class_id'].map({0: 0, 1: 1, 2: 1})

    for idx, (train_index, test_index) in enumerate(
            skf.split(train_data, train_data['class_id']),
            1
    ):
        log.info(f'Starting fold {idx} / {skf.n_splits}')

        net = sb.load_model(
            'efficientnet_b7',
            device=device_name,
            requires_grad=False
        )
        log.info("Done.")

        criterion, criterion_val = sb.get_loss(config, device=device_name)

        optimizer = sb.get_optimizer(config, config.train.optimizer, net)

        scheduler = sb.get_scheduler(config, optimizer)

        train_epoch = tqdm(
            #range(config.train.n_epoch),
            range(15),
            dynamic_ncols=True,
            desc='Epochs',
            position=0
        )

        train_labels, valid_labels = train_data.loc[train_index], \
                                     train_data.loc[test_index]

        log.info(train_labels['class_id'].value_counts())
        log.info(valid_labels['class_id'].value_counts())

        dt, dv = sb.get_data_loaders(
            train_labels,
            valid_labels,
            config,
            TRAIN_PHOTOS_DIR,
            TRAIN_PHOTOS_DIR
        )

        log.info(os.listdir(str(DATA_DIR)))

        out_dir = str(ROOT_DIR / config.outdir)
        log.info("Savedir: {}".format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        f1_best = - float("inf")

        for epoch in train_epoch:
            sb.train(net, dt, criterion, optimizer, config, epoch)
            f1_avg = sb.validation(net, dv, criterion_val, epoch)
            if f1_avg >= f1_best:
                filename = str(
                    WEIGHTS_DIR / 'step_1' / f'ms_{idx}_step_1.pth'
                )
                sb.save_checkpoint(net, optimizer,
                                   scheduler, epoch,
                                   filename, f1_avg)
                f1_best = f1_avg
            scheduler.step(f1_avg)

    log.info('END TRAINING MODEL FRO STEP 1')


def train_model_step_3():
    log.info('TRAINING MODEL FRO STEP 1')
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    config = sb.convert_dict_to_tuple(dictionary=data)

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device_name)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id

    log.info(f'device: {device_name}')

    log.info("Loading model...")

    skf = StratifiedKFold(n_splits=3)

    train_data = pd.read_csv(TRAIN_LABELS_DIR, sep='\t')

    for idx, (train_index, test_index) in enumerate(
            skf.split(train_data, train_data['class_id']),
            5
    ):
        log.info(f'Starting fold {idx} / {skf.n_splits}')

        net = sb.load_model(
            'efficientnet_b6',
            device=device_name,
            requires_grad=False
        )
        log.info("Done.")

        criterion, criterion_val = sb.get_loss(config, device=device_name)

        optimizer = sb.get_optimizer(config, config.train.optimizer, net)

        scheduler = sb.get_scheduler(config, optimizer)

        train_epoch = tqdm(
            #range(config.train.n_epoch),
            range(15),
            dynamic_ncols=True,
            desc='Epochs',
            position=0
        )

        train_labels, valid_labels = train_data.loc[train_index], \
                                     train_data.loc[test_index]

        log.info(train_labels['class_id'].value_counts())
        log.info(valid_labels['class_id'].value_counts())

        dt, dv = sb.get_data_loaders(
            train_labels,
            valid_labels,
            config,
            TRAIN_PHOTOS_DIR,
            TRAIN_PHOTOS_DIR
        )

        log.info(os.listdir(str(DATA_DIR)))

        out_dir = str(ROOT_DIR / config.outdir)
        log.info("Savedir: {}".format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        f1_best = - float("inf")

        for epoch in train_epoch:
            sb.train(net, dt, criterion, optimizer, config, epoch)
            f1_avg = sb.validation(net, dv, criterion_val, epoch)
            if f1_avg >= f1_best:
                filename = str(
                    WEIGHTS_DIR / 'step_1' / f'ms_{idx}_step_1.pth'
                )
                sb.save_checkpoint(net, optimizer,
                                   scheduler, epoch,
                                   filename, f1_avg)
                f1_best = f1_avg
            scheduler.step(f1_avg)

    log.info('END TRAINING MODEL FRO STEP 1')


def train_model_step_2():
    log.info('TRAINING MODEL FRO STEP 2')
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    config = sb.convert_dict_to_tuple(dictionary=data)

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device_name)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id

    log.info(f'device: {device_name}')

    log.info("Loading model...")

    skf = StratifiedKFold(n_splits=3)

    train_data = pd.read_csv(TRAIN_LABELS_DIR, sep='\t')
    train_data = train_data[train_data['class_id'].isin([1, 2])]
    train_data['class_id'] = train_data['class_id'].map({1: 0, 2: 1})

    for idx, (train_index, test_index) in enumerate(
            skf.split(train_data, train_data['class_id']),
            1
    ):
        log.info(f'Starting fold {idx} / {skf.n_splits}')

        net = sb.load_model(
            config,
            device=device_name,
            requires_grad=False
        )
        log.info("Done.")

        criterion, criterion_val = sb.get_loss(config, device=device_name)

        optimizer = sb.get_optimizer(config, config.train.optimizer, net)

        scheduler = sb.get_scheduler(config, optimizer)

        train_epoch = tqdm(
            #range(config.train.n_epoch),
            range(18),
            dynamic_ncols=True,
            desc='Epochs',
            position=0
        )

        train_labels, valid_labels = train_data.loc[train_index], \
                                     train_data.loc[test_index]

        log.info(train_labels['class_id'].value_counts())
        log.info(valid_labels['class_id'].value_counts())
        log.info(len(train_labels))
        log.info(len(valid_labels))

        dt, dv = sb.get_data_loaders(
            train_labels,
            valid_labels,
            config,
            TRAIN_PHOTOS_DIR,
            TRAIN_PHOTOS_DIR
        )

        log.info(os.listdir(str(DATA_DIR)))

        out_dir = str(ROOT_DIR / config.outdir)
        log.info("Savedir: {}".format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        f1_best = - float("inf")

        for epoch in train_epoch:
            sb.train(net, dt, criterion, optimizer, config, epoch)
            f1_avg = sb.validation(net, dv, criterion_val, epoch)
            if f1_avg >= f1_best:
                filename = str(
                    WEIGHTS_DIR / 'step_2' / f'ms_{idx}_step_2.pth'
                )
                sb.save_checkpoint(net, optimizer,
                                   scheduler, epoch,
                                   filename, f1_avg)
                f1_best = f1_avg
            scheduler.step(f1_avg)

    log.info('END TRAINING MODEL FRO STEP 2')


def train_model():
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    config = sb.convert_dict_to_tuple(dictionary=data)

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device_name)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id

    log.info(f'device: {device_name}')

    log.info("Loading model...")
    net = sb.load_model(
        config,
        device=device_name,
        requires_grad=False
    )
    log.info("Done.")

    criterion, criterion_val = sb.get_loss(config, device=device_name)

    optimizer = sb.get_optimizer(config, config.train.optimizer, net)

    scheduler = sb.get_scheduler(config, optimizer)

    train_epoch = tqdm(
        range(config.train.n_epoch),
        dynamic_ncols=True,
        desc='Epochs',
        position=0
    )

    train_data = pd.read_csv(TRAIN_LABELS_DIR, sep='\t')

    train_labels, valid_labels = train_test_split(
        train_data,
        stratify=train_data['class_id'],
        test_size=0.05,
        random_state=seed
    )
    log.info(train_labels['class_id'].value_counts())
    log.info(valid_labels['class_id'].value_counts())

    dt, dv = sb.get_data_loaders(
        train_labels,
        valid_labels,
        config,
        TRAIN_PHOTOS_DIR,
        TRAIN_PHOTOS_DIR
    )

    log.info(os.listdir(str(DATA_DIR)))

    out_dir = str(ROOT_DIR / config.outdir)
    log.info("Savedir: {}".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f1_best = - float("inf")

    for epoch in train_epoch:
        sb.train(net, dt, criterion, optimizer, config, epoch)
        f1_avg = sb.validation(net, dv, criterion_val, epoch)
        if f1_avg >= f1_best:
            sb.save_checkpoint(net, optimizer,
                               scheduler, epoch,
                               out_dir, f1_avg)
            f1_best = f1_avg
        scheduler.step(f1_avg)


def train_model_2():
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    config = sb.convert_dict_to_tuple(dictionary=data)

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device_name)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id

    log.info(f'device: {device_name}')

    log.info("Loading model...")
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    train_data = pd.read_csv(TRAIN_LABELS_DIR, sep='\t')
    
    for idx, (train_index, test_index) in enumerate(
        skf.split(
            train_data, 
            train_data['class_id']
        ),
        1
    ):
        log.info(f'Starting fold {idx} / {skf.n_splits}')
    
        net = sb.load_model(
            config,
            device=device_name,
            requires_grad=False
        )
        log.info("Done.")

        criterion, criterion_val = sb.get_loss(config, device=device_name)

        optimizer = sb.get_optimizer(config, config.train.optimizer, net)

        scheduler = sb.get_scheduler(config, optimizer)

        train_epoch = tqdm(
            range(config.train.n_epoch),
            dynamic_ncols=True,
            desc='Epochs',
            position=0
        )
        
        train_labels, valid_labels = train_data.loc[train_index], train_data.loc[test_index]

        log.info(train_labels['class_id'].value_counts())
        log.info(valid_labels['class_id'].value_counts())
        
        dt, dv = sb.get_data_loaders(
            train_labels,
            valid_labels,
            config,
            TRAIN_PHOTOS_DIR,
            TRAIN_PHOTOS_DIR
        )
        
        log.info(os.listdir(str(DATA_DIR)))
        
        out_dir = str(ROOT_DIR / config.outdir)
        log.info("Savedir: {}".format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        f1_best = - float("inf")

        best_model = None

        for epoch in train_epoch:
            sb.train(net, dt, criterion, optimizer, config, epoch)
            f1_avg = sb.validation(net, dv, criterion_val, epoch)
            if f1_avg > f1_best:
                sb.save_checkpoint(net, optimizer,
                                scheduler, epoch,
                                out_dir, f1_avg, idx)
                f1_best = f1_avg
                best_model = net
            scheduler.step()


    
    # train_labels, valid_labels = train_test_split(
    #     train_data,
    #     stratify=train_data['class_id'],
    #     test_size=0.15,
    #     random_state=seed
    # )
    
    # sb.set_requires_grad(best_model, True)
    #
    # sb.train(best_model, dt, criterion, optimizer, config, 0)
    # f1_avg = sb.validation(net, dv, criterion_val, epoch)
    # if f1_avg > f1_best:
    #     sb.save_checkpoint(net, optimizer,
    #                        scheduler, epoch,
    #                        out_dir, f1_avg)
    #     f1_best = f1_avg


# темплейт для запуска тренировки модели
if __name__ == "__main__":
    train_model()
    log.info("model trained")
