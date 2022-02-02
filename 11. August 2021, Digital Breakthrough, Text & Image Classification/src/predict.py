import yaml
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.preprocess import preprocess
from src.metrics import f1_score
from definitions import (
    DATA_PATH,
    DESCRIPTION_PATH,
    PREDICTION_CONFIGS_PATH
)


def prediction(config_name: str,
               agg='sum',
               test=None,
               cat=0):
    with open(PREDICTION_CONFIGS_PATH / config_name) as f:
        input_dict = yaml.safe_load(f)
    if test is None:
        test = pd.read_csv(DATA_PATH / 'test.csv')
    if cat == 0:
        categories = (
            pd.read_csv(
                DESCRIPTION_PATH / 'categories.csv')['category'].tolist()
        )
    else:
        categories = (
            pd.read_csv(
                DESCRIPTION_PATH / 'categories2.csv')['category'].tolist()
        )

    test_ = test[~(test.description.isna())]
    pb = []
    item_name = test_.description
    for k, v in input_dict.items():
        model = v['model']
        tokenizer = v['tokenizer']
        inp = preprocess(item_name,
                         tokenizer_path=tokenizer)
        mdl = load_model(model,
                         custom_objects={'f1_score': f1_score})
        p = mdl.predict(inp, batch_size=256, verbose=True)
        pb.append(p)
    if agg == 'mean':
        total_array = np.mean(np.array(pb), axis=0)
    elif agg == 'median':
        total_array = np.median(np.array(pb), axis=0)
    elif agg == 'sum':
        total_array = np.sum(np.array(pb), axis=0)
    else:
        raise ValueError(f'unknown agg: {agg}')
    df_pb = pd.DataFrame(total_array, columns=categories)
    pred = df_pb.idxmax(axis=1)
    pred.index = test_.index
    return pred
