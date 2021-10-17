import os

import numpy as np
from tqdm import tqdm

from . import mnist

path = os.path.dirname(os.path.realpath(__file__))


def split():

    train_x, train_y, val_x, val_y = mnist.load()

    for i in tqdm(range(10)):
        train = train_x[np.where(train_y == i)]
        val = val_x[np.where(val_y == i)]
        train = train.reshape(len(train), 28 * 28)
        val = val.reshape(len(val), 28 * 28)
        fn1 = "mnist_train_" + str(i) + ".csv"
        fn2 = "mnist_val_" + str(i) + ".csv"
        np.savetxt(os.path.join(path, fn1), train, delimiter=",")
        np.savetxt(os.path.join(path, fn2), val, delimiter=",")


if __name__ == "__main__":
    split()
