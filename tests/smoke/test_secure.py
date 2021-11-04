from threading import Thread

import pytest
from singa import tensor

from src.client.app import Client
from src.server.app import Server

from ..helpers.run_command import run_command
from ..helpers.start_app import max_epoch


@pytest.mark.parametrize("num_clients", [3, 5])
def test_communication(num_clients):
    # Init
    s = Server(num_clients=num_clients, secure=True)
    s._Server__start_connection()

    thread_c = []
    for i in range(num_clients):
        t = run_command(
            f"python -m tests.helpers.start_app --mode client --global_rank {i} --secure",
            thread=True,
        )
        thread_c.append(t)

    s._Server__start_rank_pairing()
    s._Server__init_secure_aggregation()

    # Training
    s.pull()
    for _ in range(max_epoch):
        s.push()
        s.pull()

    for i in range(num_clients):
        thread_c[i].join()

    s.close()


@pytest.mark.parametrize("num_clients", [3, 5])
def test_secure_aggregation(num_clients):
    weights = {}
    for i in range(3):
        weights["w" + str(i)] = tensor.random((3, 3))

    # Without secure
    s = Server(num_clients=num_clients)
    s._Server__start_connection()

    thread_c = []
    for i in range(num_clients):
        t = run_command(
            f"python -m tests.helpers.start_app --mode client --global_rank {i}", thread=True
        )
        thread_c.append(t)

    s._Server__start_rank_pairing()
    s.pull()
    s.weights = weights
    for _ in range(max_epoch):
        s.push()
        s.pull()

    for i in range(num_clients):
        thread_c[i].join()

    weights_without_secure = s.weights
    s.close()

    # With secure
    s = Server(num_clients=num_clients, secure=True, port=4321)
    s._Server__start_connection()

    thread_c = []
    for i in range(num_clients):
        t = run_command(
            f"python -m tests.helpers.start_app --mode client --global_rank {i} --secure --port 4321",
            thread=True,
        )
        thread_c.append(t)

    s._Server__start_rank_pairing()
    s._Server__init_secure_aggregation()
    s.pull()
    s.weights = weights
    for _ in range(max_epoch):
        s.push()
        s.pull()

    for i in range(num_clients):
        thread_c[i].join()

    weights_with_secure = s.weights
    s.close()

    # Compare
    assert weights_without_secure == weights_with_secure
