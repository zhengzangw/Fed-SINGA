#!/usr/bin/env python3
import argparse
import socket
import struct

import numpy as np
from google.protobuf.message import Message
from singa import tensor

from ..proto import interface_pb2 as proto
from ..proto import utils


class Client:
    """Client sends and receives protobuf messages"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = 1234,
        pack_format: str = "Q",
        global_rank: int = 1,
    ) -> None:
        """Class init method

        Args:
            host (str, optional): Host ip address. Defaults to '127.0.0.1'.
            port (str, optional): Port. Defaults to 1234.
        """
        self.host = host
        self.port = port
        self.sock = socket.socket()

        self.global_rank = global_rank

        self.pack_format = pack_format

    def init_weights(self):
        self.weights = tensor.from_numpy(np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32))

    def start(self) -> None:
        self.sock.connect((self.host, self.port))
        utils.send_int(self.sock, self.global_rank)

        print(f"[Client {self.global_rank}] Connect to {self.host}:{self.port}")

    def pull(self):
        message = proto.WeightsExchange()
        utils.receive_message(self.sock, message, self.pack_format)
        weights = utils.deserialize_tensor(message.weights)
        self.weights = weights

    def push(self) -> None:
        message = proto.WeightsExchange()
        message.op_type = proto.PUSH
        message.weights = utils.serialize_tensor(self.weights)

        utils.send_message(self.sock, message, self.pack_format)

    def close(self) -> None:
        self.sock.close()


def test(global_rank=0):
    client = Client(global_rank=global_rank)
    client.start()
    client.init_weights()

    max_epoch = 3
    for i in range(max_epoch):
        print(f"On epoch {i}:")

        # Pull from Server
        client.pull()
        print(client.weights)

        # Update locally
        client.weights += global_rank + 1

        # Push to Server
        client.push()
        print(client.weights)

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_rank", default=0, type=int)
    args = parser.parse_args()

    test(args.global_rank)
