import json
import os

import numpy as np
from PIL import Image


def get_data(train_split=0.7, val_split=0.2):
    # get list of file name
    dataset = os.listdir("dataset")
    file_names = [file[:-4] for file in dataset if file[-3:] == "jpg"]

    # empty arrays for images/labels
    data_size = len(file_names)
    images = np.zeros((data_size, 100, 100, 3))
    labels = np.zeros((data_size, 16))
    for i, file in enumerate(file_names):
        images[i] = np.asarray(Image.open('dataset/{}.jpg'.format(file)))

        # parse label
        label_file = open('dataset/{}.json'.format(file))
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