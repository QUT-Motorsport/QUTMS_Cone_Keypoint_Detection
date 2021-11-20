import json
import os

import numpy as np
from PIL import Image


def get_data(train_split=0.7, val_split=0.2, data_path='dataset/striped'):
    # get list of file name
    dataset = os.listdir(data_path)
    file_names = [file[:-5] for file in dataset if file[-4:] == "json"]

    # empty arrays for images/labels
    data_size = len(file_names)
    images = np.zeros((data_size, 100, 100, 3))
    labels = np.zeros((data_size, 16))
    for i, file in enumerate(file_names):
        images[i] = np.asarray(Image.open('{}/{}.jpg'.format(data_path, file)))

        # parse label
        try:
            label_file = open('{}/{}.json'.format(data_path, file))
        except:
            continue
        label_data = json.load(label_file)
        label_file.close()
        # assert label
        assert len(label_data["shapes"]) == 1
        # sort label
        keypoints = np.array(label_data["shapes"][0]["points"])
        order = keypoints[:, 0].argsort()
        # shape label
        labels[i] = keypoints[order, :].reshape(16)

    # scale down to 0-1 range
    images = images.astype('float32') / 255
    labels = labels.astype('float32') / 1

    # train/validation/test splits
    train_index = int(data_size * train_split)
    val_split = int(data_size * val_split)

    X = images[:train_index]
    x_val = images[train_index:train_index + val_split]
    x_test = images[train_index + val_split:]

    Y = labels[:train_index]
    y_val = labels[train_index:train_index + val_split]
    y_test = labels[train_index + val_split:]

    return X, x_val, x_test, Y, y_val, y_test