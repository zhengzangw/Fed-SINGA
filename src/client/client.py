#!/usr/bin/env python3
import argparse
import os
import socket
import time

import numpy as np
from PIL import Image
from singa import device, opt, tensor
from tqdm import tqdm

from .app import Client
from .data import cifar10, cifar100, mnist
from .model import alexnet, cnn, mlp, resnet, xceptionnet

np_dtype = {"float16": np.float16, "float32": np.float32}
singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


# Data augmentation
def augmentation(x, batch_size):
    xpad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], "symmetric")
    for data_num in range(0, batch_size):
        offset = np.random.randint(8, size=2)
        x[data_num, :, :, :] = xpad[
            data_num, :, offset[0] : offset[0] + x.shape[2], offset[1] : offset[1] + x.shape[2]
        ]
        if_flip = np.random.randint(2)
        if if_flip:
            x[data_num, :, :, :] = x[data_num, :, :, ::-1]
    return x


# Calculate accuracy
def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


# Data partition according to the rank
def partition(global_rank, world_size, train_x, train_y, val_x, val_y):
    # Partition training data
    data_per_rank = train_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    train_x = train_x[idx_start:idx_end]
    train_y = train_y[idx_start:idx_end]

    # Partition evaluation data
    data_per_rank = val_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    val_x = val_x[idx_start:idx_end]
    val_y = val_y[idx_start:idx_end]
    return train_x, train_y, val_x, val_y


# Function to all reduce NUMPY accuracy and loss from multiple devices
def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    dist_opt.all_reduce(reducer.data)
    dist_opt.wait()
    output = tensor.to_numpy(reducer)
    return output


def load_mnist_dataset(num=0):
    print("loading non-iid mnist dataset ", num)
    fn1 = "data/mnist_train_" + str(num) + ".csv"
    fn2 = "data/mnist_val_" + str(num) + ".csv"
    train = np.loadtxt(fn1, delimiter=",")
    val = np.loadtxt(fn2, delimiter=",")
    train = train.reshape(len(train), 1, 28, 28)
    val = val.reshape(len(val), 1, 28, 28)
    return (
        train,
        np.full((len(train)), num, dtype=np.int32),
        val,
        np.full((len(val)), num, dtype=np.int32),
    )


def resize_dataset(x, image_size):
    num_data = x.shape[0]
    dim = x.shape[1]
    X = np.zeros(shape=(num_data, dim, image_size, image_size), dtype=np.float32)
    for n in range(0, num_data):
        for d in range(0, dim):
            X[n, d, :, :] = np.array(
                Image.fromarray(x[n, d, :, :]).resize((image_size, image_size), Image.BILINEAR),
                dtype=np.float32,
            )
    return X


def get_data(data, data_dist="iid", device_id=None):
    if data == "cifar10":
        train_x, train_y, val_x, val_y = cifar10.load()
        num_classes = 10
    elif data == "cifar100":
        train_x, train_y, val_x, val_y = cifar100.load()
        num_classes = 100
    elif data == "mnist":
        if data_dist == "iid":
            train_x, train_y, val_x, val_y = mnist.load()
        else:
            train_x, train_y, val_x, val_y = load_mnist_dataset(device_id)
        num_classes = 10
    return train_x, train_y, val_x, val_y, num_classes


def get_model(model, num_channels=None, num_classes=None, data_size=None):
    if model == "resnet":
        model = resnet.resnet50(num_channels=num_channels, num_classes=num_classes)
    elif model == "xceptionnet":
        model = xceptionnet.create_model(num_channels=num_channels, num_classes=num_classes)
    elif model == "cnn":
        model = cnn.create_model(num_channels=num_channels, num_classes=num_classes)
    elif model == "alexnet":
        model = alexnet.create_model(num_channels=num_channels, num_classes=num_classes)
    elif model == "mlp":
        model = mlp.create_model(data_size=data_size, num_classes=num_classes)
    else:
        raise NotImplementedError
    return model


def run(
    global_rank,
    world_size,
    device_id,
    max_epoch,
    batch_size,
    model,
    data,
    data_dist,
    sgd,
    graph,
    verbosity,
    dist_option="plain",
    spars=None,
    precision="float32",
):

    client = Client(global_rank=device_id)
    client.start()
    client.init_weights()

    dev = device.get_default_device()
    dev.SetRandSeed(0)
    np.random.seed(0)

    # Prepare dataset
    train_x, train_y, val_x, val_y, num_classes = get_data(data, data_dist, device_id)
    train_x_, train_y_, val_x_, val_y_, num_classes_ = get_data(data, "iid", device_id)

    num_channels = train_x.shape[1]
    image_size = train_x.shape[2]
    data_size = np.prod(train_x.shape[1 : train_x.ndim]).item()

    # Prepare model
    model = get_model(
        model, num_channels=num_channels, num_classes=num_classes, data_size=data_size
    )

    # For distributed training, sequential has better performance
    if hasattr(sgd, "communicator"):
        DIST = True
        sequential = True
    else:
        DIST = False
        sequential = False

    if DIST:
        train_x, train_y, val_x, val_y = partition(
            global_rank, world_size, train_x, train_y, val_x, val_y
        )
    """
    # point dataset shape correctness
    if global_rank == 0:
        print("Check the shape of dataset:")
        print(train_x.shape)
        print(train_y.shape)
    """

    if model.dimension == 4:
        tx = tensor.Tensor(
            (batch_size, num_channels, model.input_size, model.input_size),
            dev,
            singa_dtype[precision],
        )
    elif model.dimension == 2:
        tx = tensor.Tensor((batch_size, data_size), dev, singa_dtype[precision])
        np.reshape(train_x, (train_x.shape[0], -1))
        np.reshape(val_x, (val_x.shape[0], -1))

    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    num_train_batch = train_x.shape[0] // batch_size
    num_val_batch = val_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    # Attach model to graph
    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=graph, sequential=sequential)
    dev.SetVerbosity(verbosity)

    # if data_dist == "non-iid":
    #     celtral_model_path = "checkpoint/central_model.zip"
    #     if os.path.exists(celtral_model_path):
    #         print("loading model from " + celtral_model_path)
    #         model.load_states(fpath=celtral_model_path)
    #     else:
    #     	# Pull from Server
    #     	client.pull()
    #         print("initiating a central model...")
    #         model.save_states(celtral_model_path)
    #         return

    # Training and evaluation loop
    for epoch in range(max_epoch):
        start_time = time.time()
        np.random.shuffle(idx)

        if epoch > 0:
            client.pull()
            model.set_states(client.weights)

        if global_rank == 0:
            print("Starting Epoch %d:" % (epoch))

        # Training phase
        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        # Evaluation phase
        model.eval()
        for b in range(num_val_batch):
            x = val_x[b * batch_size : (b + 1) * batch_size]
            if model.dimension == 4:
                if image_size != model.input_size:
                    x = resize_dataset(x, model.input_size)
            x = x.astype(np_dtype[precision])
            y = val_y[b * batch_size : (b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            out_test = model(tx)
            test_correct += accuracy(tensor.to_numpy(out_test), y)

        # if DIST:
        #     # Reduce the evaulation accuracy from multiple devices
        #     test_correct = reduce_variable(test_correct, sgd, reducer)

        # Output the evaluation accuracy
        if global_rank == 0:
            print(
                "Evaluation accuracy = %f, Elapsed Time = %fs"
                % (
                    test_correct / (num_val_batch * batch_size * world_size),
                    time.time() - start_time,
                )
            )

        model.train()
        for b in tqdm(range(num_train_batch)):
            # Generate the patch data in this iteration
            x = train_x[idx[b * batch_size : (b + 1) * batch_size]]
            if model.dimension == 4:
                x = augmentation(x, batch_size)
                if image_size != model.input_size:
                    x = resize_dataset(x, model.input_size)
            x = x.astype(np_dtype[precision])
            y = train_y[idx[b * batch_size : (b + 1) * batch_size]]

            # Copy the patch data into input tensors
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            # Train the model
            out, loss = model(tx, ty, dist_option, spars)
            train_correct += accuracy(tensor.to_numpy(out), y)
            train_loss += tensor.to_numpy(loss)[0]

        if DIST:
            # Reduce the evaluation accuracy and loss from multiple devices
            reducer = tensor.Tensor((1,), dev, tensor.float32)
            train_correct = reduce_variable(train_correct, sgd, reducer)
            train_loss = reduce_variable(train_loss, sgd, reducer)

        if global_rank == 0:
            print(
                "Training loss = %f, training accuracy = %f"
                % (train_loss, train_correct / (num_train_batch * batch_size * world_size))
            )

        client.weights = model.get_states()
        client.push()

    dev.PrintTimeProfiling()


def parseargs():
    # Use argparse to get command config: max_epoch, model, data, etc., for single gpu training
    parser = argparse.ArgumentParser(description="Training using the autograd and graph.")
    parser.add_argument(
        "--model", choices=["cnn", "resnet", "xceptionnet", "mlp", "alexnet"], default="mlp"
    )
    parser.add_argument("--data", choices=["mnist", "cifar10", "cifar100"], default="mnist")
    parser.add_argument("-p", choices=["float32", "float16"], default="float32", dest="precision")
    parser.add_argument(
        "-m", "--max-epoch", default=10, type=int, help="maximum epochs", dest="max_epoch"
    )
    parser.add_argument(
        "-b", "--batch-size", default=64, type=int, help="batch size", dest="batch_size"
    )
    parser.add_argument(
        "-l", "--learning-rate", default=0.005, type=float, help="initial learning rate", dest="lr"
    )
    # Determine which gpu to use
    parser.add_argument(
        "-i", "--device-id", default=0, type=int, help="which GPU to use", dest="device_id"
    )
    parser.add_argument(
        "-g",
        "--disable-graph",
        default="True",
        action="store_false",
        help="disable graph",
        dest="graph",
    )
    parser.add_argument(
        "-v", "--log-verbosity", default=0, type=int, help="logging verbosity", dest="verbosity"
    )
    parser.add_argument(
        "-d",
        "--data-distribution",
        choices=["iid", "non-iid"],
        default="iid",
        help="data distribution",
        dest="data_dist",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()

    sgd = opt.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-5, dtype=singa_dtype[args.precision])

    run(
        0,
        1,
        args.device_id,
        args.max_epoch,
        args.batch_size,
        args.model,
        args.data,
        args.data_dist,
        sgd,
        args.graph,
        args.verbosity,
        precision=args.precision,
    )
