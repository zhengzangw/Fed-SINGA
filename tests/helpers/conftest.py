import socket
from typing import Tuple

import pytest
from google.protobuf.message import Message
from singa import tensor

from src.proto import interface_pb2
from src.proto.utils import serialize_tensor


@pytest.fixture(scope="module")
def server_client() -> Tuple[socket.socket, socket.socket]:
    HOST = "127.0.0.1"
    PORT = 1234

    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen()

    c = socket.socket()
    c.connect((HOST, PORT))
    s = s.accept()[0]

    yield (s, c)

    c.shutdown(0)
    s.shutdown(0)


@pytest.fixture
def protobuf() -> Message:
    p = interface_pb2.WeightsExchange()
    p.op_type = interface_pb2.DEFAULT
    p.weights["x"] = "placeholder message".encode("utf-8")
    p.weights["y"] = serialize_tensor(tensor.random((3, 3)))
    return p
