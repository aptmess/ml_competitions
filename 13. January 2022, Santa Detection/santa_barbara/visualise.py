import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
import sys

from santa_barbara.utils import read_image
from definitions import TRAIN_PHOTOS_DIR, TRAIN_LABELS_DIR, TRAIN_IMAGES

sys.path.append('.')

def visualise():
    train_data = pd.read_csv(TRAIN_LABELS_DIR, sep='\t')

    train_data_0 = train_data[train_data['class_id'] == 0].reset_index(drop=True)
    train_data_1 = train_data[train_data['class_id'] == 1].reset_index(drop=True)
    train_data_2 = train_data[train_data['class_id'] == 2].reset_index(drop=True)

    # Figure size

    # Subplot
    for batch, k in enumerate(np.split(train_data_0, 50), 1):
        fig = plt.figure(figsize=(16, 16))
        fig.suptitle('Class 0 - Man with the beard', fontsize=20)
        for i, j in k.reset_index(drop=True).iterrows():
            img = np.asarray(read_image(str(TRAIN_PHOTOS_DIR / j['image_name'])))
            ax = plt.subplot(4, 4, i + 1)
            ax.grid(False)
            plt.imshow(img)
            plt.xlabel(j['image_name'])

        plt.savefig(f'../data/train/images/0-{batch}-50.png')

    # Subplot

    for batch, k in enumerate(np.split(train_data_1, 15), 1):
        fig = plt.figure(figsize=(16, 16))
        fig.suptitle('Class 1 - Ded Moroz', fontsize=20)
        for i, j in k.reset_index(drop=True).iterrows():
            img = np.asarray(read_image(str(TRAIN_PHOTOS_DIR / j['image_name'])))
            ax = plt.subplot(4, 4, i + 1)
            ax.grid(False)
            plt.imshow(img)
            plt.xlabel(j['image_name'])

        plt.savefig(f'../data/train/images/1-{batch}-15.png')

    for batch, k in enumerate(np.split(train_data_2, 15), 1):
        fig = plt.figure(figsize=(16, 16))
        fig.suptitle('Class 2 - Santa Claus', fontsize=20)
        for i, j in k.reset_index(drop=True).iterrows():
            img = np.asarray(read_image(str(TRAIN_PHOTOS_DIR / j['image_name'])))
            ax = plt.subplot(4, 4, i + 1)
            ax.grid(False)
            plt.imshow(img)
            plt.xlabel(j['image_name'])

        plt.savefig(f'../data/train/images/2-{batch}-15.png')


if __name__ == '__main__':
    visualise()
