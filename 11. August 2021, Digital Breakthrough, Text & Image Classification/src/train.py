"""
Training model on all data.
"""
import os
import yaml
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
from src.metrics import f1_score
from src.models import distilbert_model
from src.preprocess import preprocess
from definitions import (
    CONFIGS_PATH,
    TOKENIZERS_PATH,
    LM_MODELS_PATH,
    CLASSIFICATION_MODELS_PATH
)
from src.prepare_data import prepare_data

from src.train_lm import (
    LMTrainer,
    LMModel,
    LMDataset,
    LMDataCollator,
    LMTokenizer,
    LMTrainingArgs
)
from src.train_tokenizers import TokenizerFabrica

os.environ['WANDB_DISABLED'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(
    device=tf.config.experimental.get_visible_devices('GPU')[0],
    enable=True
)

seed = 42
np.random.seed(seed)
random.seed(seed)


def train_description_model(config_name='config_distilbert.yaml',
                            skip_fitting_tokenizer=False,
                            skip_fitting_lm_model=False,
                            skip_fitting_classification_model=False):
    with open(CONFIGS_PATH / config_name) as f:
        data = yaml.safe_load(f)

    for model_name, model_dict in data.items():

        path = model_dict['paths']

        path_input_data = path['data_names_path']
        path_item_name = path['item_names_path']
        path_checkpoint_name = path['path_checkpoint']
        path_name = path['path_name']
        path_categories = path['categories_name']
        path_tr = path['path_tr']
        path_valid = path['path_valid']
        valid_split = path['valid_split']
        if path.get('skip_typology'):
            path_skip_typology = path['skip_typology']
        else:
            path_skip_typology = []

        save = model_dict['save_names']
        save_cl_model = save['classification_model_name']
        save_token_name = save['tokenizer_name']


        args = model_dict['args']

        args_lm_names = args['language_model']
        args_lm_names_model_name = args_lm_names['model_name']
        args_lm_names_model = args_lm_names['model']
        args_lm_names_training_args = args_lm_names['training_args']
        args_lm_names_data_collator = args_lm_names['data_collator']
        args_lm_names_dataset = args_lm_names['dataset']

        args_tokenizers = args['tokenizers']
        args_tokenizers_pretrained = args_tokenizers.get('pretrained', False)
        args_classification_model = args['classification_model']
        args_classification_model_max_len = args_classification_model['max_len']
        args_classification_model_train_arg = args_classification_model['train']
        args_classification_model_exp = args_classification_model.get(
            'experiment',
            False
        )

        output_path_name = f"{save_cl_model}_{save_token_name.split('.')[0]}"

        output_path_tokenizer = TOKENIZERS_PATH / save_token_name
        output_path_lm_models = LM_MODELS_PATH / output_path_name

        save_model_name = save.get('save_model_name', output_path_name)
        print(save_model_name)
        output_path_model = CLASSIFICATION_MODELS_PATH / save_model_name

        path_item, path_item_train, path_item_test, path_input, len_cat = (
            prepare_data(
                path_name,
                path_item_name,
                path_input_data,
                path_categories,
                name_tr=path_tr,
                name_valid=path_valid,
                validation_split=valid_split,
                skip_typology=path_skip_typology
            )
        )
        print(path_item, path_item_train, path_item_test, path_input, len_cat)

        with open(path_item, 'r') as f:
            items = f.readlines()

        if not skip_fitting_tokenizer:
            token = TokenizerFabrica(**args_tokenizers)
            token.fit(item_names=items)
            token.save_model(output_path=output_path_tokenizer)

        print(output_path_tokenizer)
        tokenizer = LMTokenizer(
            tokenizer_path=str(output_path_tokenizer)
        )

        if not skip_fitting_lm_model:
            lm = LMTrainer(
                model=LMModel(
                    model_name=args_lm_names_model_name,
                    **args_lm_names_model
                ),

                training_args=LMTrainingArgs(
                    output_dir=str(output_path_lm_models),
                    **args_lm_names_training_args
                ),

                data_collator=LMDataCollator(
                    tokenizer=tokenizer,
                    **args_lm_names_data_collator
                ),

                train_dataset=LMDataset(
                    tokenizer=tokenizer,
                    file_path=path_item_train,
                    **args_lm_names_dataset
                ),
                eval_dataset=LMDataset(
                    tokenizer=tokenizer,
                    file_path=path_item_test,
                    **args_lm_names_dataset
                ),
            )
            lm.fit()
            lm.save_model(
                output_path=output_path_lm_models,
                name=path_checkpoint_name
            )

        if not skip_fitting_classification_model:
            data = pd.read_csv(path_input)
            print(output_path_tokenizer)
            X = preprocess(
                texts=data.description,
                tokenizer_path=str(output_path_tokenizer),
                max_len=args_classification_model_max_len)
            y = pd.get_dummies(data.typology)
            model = distilbert_model(
                model_name=args_lm_names_model_name,
                output_shape=len_cat,
                input_shape=args_classification_model_max_len,
                transformer_model=(
                    str(output_path_lm_models / path_checkpoint_name)
                ),
                metrics=[f1_score],
                experiment=args_classification_model_exp
            )
            print(model.summary())
            model.fit(X, y, **args_classification_model_train_arg)
            model.save(str(output_path_model))
