import argparse
import os

import numpy as np
from tqdm import tqdm

from . import mnist


def split(path="."):

    train_x, train_y, val_x, val_y = mnist.load()

    for i in tqdm(range(10)):
        train = train_x[np.where(train_y == i)]
        val = val_x[np.where(val_y == i)]
        train = train.reshape(len(train), 28 * 28)
        val = val.reshape(len(val), 28 * 28)
        fn1 = "data/cifar10/train_" + str(i) + ".csv"
        fn2 = "data/cifar10/val_" + str(i) + ".csv"
        np.savetxt(os.path.join(path, fn1), train, delimiter=",")
        np.savetxt(os.path.join(path, fn2), val, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="data")
    args = parser.parse_args()
    split(path=args.dir)
