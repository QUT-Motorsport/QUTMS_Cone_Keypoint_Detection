from itertools import permutations
from math import inf, sqrt
from pprint import pprint
from keras import backend as K

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import numpy as np
import tensorflow as tf


def is_this_loss(true, pred):
    pred_np = pred.numpy()
    true_np = true.numpy()
    loss = 0
    #print(pred_np)

    count = 0
    for j in range(1, pred_np.shape[0]):
        count = count + 1
        pred_y = pred_np[j, :]
        true_y = true_np[j, :]
        for i in range(0, 15, step=2):
            loss = loss + K.sqrt(K.square(pred_y[i] - true_y[i]) + K.square(pred_y[i+1] - true_y[i+1]))

    # squares = K.square(true - pred)
    # Cs = K.sum(squares, axis=)
    return K.mean(K.square(pred-true)) + (loss/count) # K.mean(squares) +
    # true = true.numpy().reshape((len(true), 8, 2))
    # pred = pred.numpy().reshape((len(pred), 8, 2))
    #
    # ret = np.mean(np.mean(np.square(true-pred), axis=-1), axis=-1)
    # print(len(ret))
    # return ret

