import os
import yaml
import torch
import random
import logging
import pandas as pd
import numpy as np
import torch.backends.cudnn
import santa_barbara as sb
from tqdm import tqdm

from definitions import (
    CONFIG_PATH,
    TEST_DIR,
    OUT_DIR,
    WEIGHTS_DIR
)
from train import train_model, train_model_step_1, train_model_step_3

log = logging.getLogger(__name__)


def run_model_two_step():
    train_model_step_1()
    train_model_step_3()
    # train_model_step_2()

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

    scores = {}

    for idx, model_type in zip(
            range(1, 8),
            ['efficientnet_b7'] * 4 + ['efficientnet_b6'] * 3
    ):
        model = sb.load_resnet(
            path=str(WEIGHTS_DIR / 'step_1' / f'ms_{idx}_step_1.pth'),
            model_type=model_type,
            num_classes=config.dataset.num_of_classes,
            device=device_name
        )

        softmax_func = torch.nn.Softmax(dim=1)

        validation_augmentation = sb.ValidationAugmentations(config=config)

        files_test = os.listdir(TEST_DIR)

        log.info(len(files_test))

        test_epoch = tqdm(
            files_test,
            dynamic_ncols=True,
            desc='Prediction',
            position=0
        )
        scores[f'model_{idx}'] = {}

        for name in test_epoch:
            img_path = str(TEST_DIR / name)
            img = sb.read_image(img_path)
            crop, _ = validation_augmentation(
                image=img,
                annotation=None
            )
            crop = sb.PREPROCESS(crop).unsqueeze(0)
            crop = crop.to(device_name)
            out = model(crop)
            out = softmax_func(out).squeeze().detach().cpu().numpy()
            scores[f'model_{idx}'][name] = np.argmax(out)

    output_data = pd.DataFrame(scores)

    for i, row in output_data.iterrows():
        log.info(row)
    output_data = output_data.mode(axis=1)[0].astype(int).to_frame().reset_index()
    output_data.columns = ['image_name', 'class_id']

    # output_data = pd.DataFrame(
    #   scores.items(),
    #   columns=['image_name', 'class_id']
    # )
    output_data.to_csv(
        OUT_DIR / 'submission.csv',
        index=False,
        sep='\t'
    )



def run_model():
    train_model()
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

    model = sb.load_resnet(
        path=config.weights.classification_model_path,
        model_type=config.model.model_type,
        num_classes=config.dataset.num_of_classes,
        device=device_name
    )

    softmax_func = torch.nn.Softmax(dim=1)

    validation_augmentation = sb.ValidationAugmentations(config=config)

    files_test = os.listdir(TEST_DIR)

    log.info(len(files_test))

    scores = {}

    test_epoch = tqdm(
        files_test,
        dynamic_ncols=True,
        desc='Prediction',
        position=0
    )

    for name in test_epoch:
        img_path = str(TEST_DIR / name)
        img = sb.read_image(img_path)
        crop, _ = validation_augmentation(
            image=img,
            annotation=None
        )
        crop = sb.PREPROCESS(crop).unsqueeze(0)
        crop = crop.to(device_name)
        out = model(crop)
        out = softmax_func(out).squeeze().detach().cpu().numpy()
        scores[name] = np.argmax(out)

    output_data = pd.DataFrame(
      scores.items(),
      columns=['image_name', 'class_id']
    )
    output_data.to_csv(
      OUT_DIR / 'submission.csv',
      index=False,
      sep='\t'
    )


def run_model_2():
    train_model()
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
    
    for idx in range(1, 4):
        model = sb.load_resnet(
            path=f'{config.weights.classification_model_path}_{idx}.pth',
            model_type=config.model.model_type,
            num_classes=config.dataset.num_of_classes,
            device=device_name
        )
    
        softmax_func = torch.nn.Softmax(dim=1)
    
        validation_augmentation = sb.ValidationAugmentations(config=config)
    
        files_test = os.listdir(TEST_DIR)
    
        log.info(len(files_test))
    
        scores = {}
    
        test_epoch = tqdm(
            files_test,
            dynamic_ncols=True,
            desc='Prediction',
            position=0
        )
        scores[f'model_{idx}'] = {}
    
        for name in test_epoch:
            img_path = str(TEST_DIR / name)
            img = sb.read_image(img_path)
            crop, _ = validation_augmentation(
                image=img,
                annotation=None
            )
            crop = sb.PREPROCESS(crop).unsqueeze(0)
            crop = crop.to(device_name)
            out = model(crop)
            out = softmax_func(out).squeeze().detach().cpu().numpy()
            scores[f'model_{idx}'][name] = np.argmax(out)

    output_data = pd.DataFrame(scores)
    log.info(output_data)
    output_data = output_data.mode(axis=1)[0].astype(int).to_frame().reset_index()
    output_data.columns = ['image_name', 'class_id']
    
    # output_data = pd.DataFrame(
    #   scores.items(),
    #   columns=['image_name', 'class_id']
    # )
    output_data.to_csv(
      OUT_DIR / 'submission.csv',
      index=False,
      sep='\t'
    )


# темплейт для запуска решения
if __name__ == "__main__":
    # run_model()
    run_model_two_step()
    log.info("Model build. Ok")
