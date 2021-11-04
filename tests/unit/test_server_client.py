from threading import Thread
from typing import Tuple

import pytest
from singa import tensor

from src.client.app import Client
from src.server.app import Server

from ..helpers.conftest import server_client_single


def test_start() -> Tuple[Server, Client]:
    server = Server(num_clients=1)
    client = Client(global_rank=0)
    thread_s = Thread(target=server.start)
    thread_s.start()
    client.start()
    thread_s.join()
    client.close()
    server.close()

    server = Server(num_clients=1)
    client = Client(global_rank=0)
    thread_c = Thread(target=client.start)
    thread_c.start()
    server.start()
    thread_c.join()
    client.close()
    server.close()


def test_single_server_client(server_client_single: Tuple[Server, Client]):
    s, c = server_client_single
    max_epoch = 3

    weights = {}
    sum_weights = 0
    for i in range(3):
        weights["w" + str(i)] = tensor.random((3, 3))
        sum_weights += tensor.to_numpy(tensor.sum(weights["w" + str(i)])).item()
    s.weights = weights

    for i in range(max_epoch):
        # scatter
        t1 = Thread(target=s.push)
        t1.start()
        t2 = Thread(target=c.pull)
        t2.start()
        t1.join()
        t2.join()

        # Update locally
        for k in c.weights.keys():
            c.weights[k] += 1

        # print(c.weights)
        # gather
        t1 = Thread(target=c.push)
        t1.start()
        t2 = Thread(target=s.pull)
        t2.start()
        t1.join()
        t2.join()

    sum_w = 0
    for i in range(3):
        sum_w += tensor.to_numpy(tensor.sum(weights["w" + str(i)])).item()
    assert max_epoch * 3 * 9 + sum_weights - sum_w < 1e-5
