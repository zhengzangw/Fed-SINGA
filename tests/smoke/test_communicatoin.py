import pytest

from src.server.app import Server

from ..helpers.run_command import run_command
from ..helpers.start_app import max_epoch


@pytest.mark.parametrize("num_clients", [1, 3, 10])
def test_communication(num_clients):
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
    for _ in range(max_epoch):
        s.push()
        s.pull()

    for i in range(num_clients):
        thread_c[i].join()

    s.close()


@pytest.mark.parametrize("num_clients", [1, 3, 10])
def test_docker_communication(num_clients):
    run_command(f"docker-compose -f scripts/docker-compose.client_{num_clients}.yml up")
