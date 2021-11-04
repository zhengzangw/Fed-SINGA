from socket import socket
from typing import Tuple

import pytest
from google.protobuf.message import Message
from singa import tensor

from src.proto import interface_pb2
from src.proto.utils import (
    deserialize_tensor,
    parseargs,
    receive_all,
    receive_int,
    receive_message,
    send_int,
    send_message,
    serialize_tensor,
)

from ..helpers.conftest import protobuf, server_client


@pytest.mark.parametrize("m", ["TEST", "A Long test txt setence."])
def test_receive_all(server_client: Tuple[socket, socket], m: str):
    s, c = server_client

    message = m.encode("utf-8")
    s.sendall(message)
    r = receive_all(c, len(message))
    recovered = r.decode("utf-8")

    assert recovered == m


@pytest.mark.parametrize("i", [10, int(1e10)])
def test_send_receive_int(server_client: Tuple[socket, socket], i: int):
    s, c = server_client
    send_int(s, i)
    assert receive_int(c) == i

    send_int(c, i)
    assert receive_int(s) == i


def test_send_receive_message(server_client: Tuple[socket, socket], protobuf: Message):
    s, c = server_client
    send_message(s, protobuf)
    p = receive_message(c, protobuf)
    assert p.op_type == interface_pb2.DEFAULT
    assert p.weights["x"].decode("utf-8") == "placeholder message"


def test_serialize():
    x = tensor.random((3, 3))
    y = deserialize_tensor(serialize_tensor(x))
    assert x == y


def test_parseargs():
    args = parseargs("")
    assert args.model == "mlp"
