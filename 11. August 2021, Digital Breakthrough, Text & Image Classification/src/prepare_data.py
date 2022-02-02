import os
import pandas as pd
from definitions import (
    DATA_PATH,
    DESCRIPTION_PATH
)
from sklearn.model_selection import train_test_split


def prepare_data(path_name,
                 item_name,
                 train_name,
                 categories_name,
                 name_tr,
                 name_valid,
                 validation_split=0.15,
                 skip_typology=None):
    if skip_typology is None:
        skip_typology = []
    OUT_DIR = DESCRIPTION_PATH / path_name
    print("Savedir: {}".format(OUT_DIR))
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    train = pd.read_csv(DATA_PATH / 'train.csv')
    train_url_only = pd.read_csv(DATA_PATH / 'train_url_only.csv')
    train_full = pd.read_csv(DATA_PATH / 'train_full_load.csv')

    train_url_only['typology'] = train_url_only.typology.replace(
        {
            'предметы прикладного искусства, быта и этнографии '
            :
            'предметы прикладного искусства, быта и этнографии'
        }
    )

    train_labels = train.typology.unique()
    typology_to_label = dict(
        zip(
            sorted(train_labels),
            range(
                len(train_labels)
            )
        )
    )

    train_url_only_train_labels = (
        train_url_only[train_url_only.typology.isin(typology_to_label.keys())]
    )

    train['url'] = 1
    full_train = pd.concat((train, train_url_only_train_labels, train_full), axis=0)
    full_train = full_train[~(full_train.typology.isna())]
    full_train = full_train[~(full_train.description.isna())]
    full_train = full_train[~(full_train.typology.isin(skip_typology))]

    item_names = train.description.drop_duplicates()
    item_names = item_names.map(lambda x: x + '\n')

    with open(OUT_DIR / item_name, 'w') as f:
        f.writelines(item_names.tolist())

    save_train = full_train.drop_duplicates('description')
    save_train = save_train[save_train.description != '']

    save_train.to_csv(OUT_DIR / train_name, index=False)

    X_train, X_valid = train_test_split(
        save_train,
        random_state=42,
        test_size=validation_split
    )

    item_names_train = X_train.description
    item_names_train = item_names_train.map(lambda x: x + '\n')
    item_names_test = X_valid.description
    item_names_test = item_names_test.map(lambda x: x + '\n')

    with open(OUT_DIR / name_tr, 'w') as f:
        f.writelines(item_names_train.tolist())

    with open(OUT_DIR / name_valid, 'w') as f:
        f.writelines(item_names_test.tolist())

    categories = sorted(save_train.typology.unique())
    categories = pd.Series(categories, name='category')
    print(categories)

    categories.to_csv(OUT_DIR / categories_name, index=False)
    return (
        OUT_DIR / item_name,
        OUT_DIR / name_tr,
        OUT_DIR / name_valid,
        OUT_DIR / train_name,
        len(categories)
    )
