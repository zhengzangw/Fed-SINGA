#!/usr/bin/env python3
import argparse
import socket

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
        self.weights = None

    def start(self) -> None:
        self.sock.connect((self.host, self.port))
        utils.send_int(self.sock, self.global_rank)

        print(f"[Client {self.global_rank}] Connect to {self.host}:{self.port}")

    def pull(self):
        message = proto.WeightsExchange()
        utils.receive_message(self.sock, message, self.pack_format)
        weights = {}
        for k, v in message.weights.items():
            weights[k] = utils.deserialize_tensor(v)
        self.weights = weights

    def push(self) -> None:
        message = proto.WeightsExchange()
        message.op_type = proto.PUSH
        for k, v in self.weights.items():
            message.weights[k] = utils.serialize_tensor(v)
        utils.send_message(self.sock, message, self.pack_format)

    def close(self) -> None:
        self.sock.close()


# class Client:
#     """Client sends and receives protobuf messages"""

#     def __init__(
#         self,
#         host: str = "127.0.0.1",
#         port: str = 1234,
#         pack_format: str = "Q",
#         global_rank: int = 1,
#     ) -> None:
#         """Class init method

#         Args:
#             host (str, optional): Host ip address. Defaults to '127.0.0.1'.
#             port (str, optional): Port. Defaults to 1234.
#         """
#         self.host = host
#         self.port = port
#         self.sock = socket.socket()

#         self.global_rank = global_rank

#         self.pack_format = pack_format

#     def init_weights(self):
#         self.weights = None

#     def start(self) -> None:
#         self.sock.connect((self.host, self.port))
#         utils.send_int(self.sock, self.global_rank)

#         print(f"[Client {self.global_rank}] Connect to {self.host}:{self.port}")

#     def pull(self):
#         message = proto.WeightsExchange()
#         utils.receive_message(self.sock, message, self.pack_format)
#         weights = {}
#         for k, v in message.weights.items():
#             weights[k] = utils.deserialize_tensor(v)
#         # weights = utils.deserialize_tensor(message.weights)
#         self.weights = weights

#     def push(self) -> None:
#         message = proto.WeightsExchange()
#         message.op_type = proto.PUSH
#         for k, v in self.weights.items():
#             message.weights[k] = utils.serialize_tensor(v)
#         # message.weights = utils.serialize_tensor(self.weights)
#         utils.send_message(self.sock, message, self.pack_format)

#     def close(self) -> None:
#         self.sock.close()


def test(global_rank=0):
    client = Client(global_rank=global_rank)
    client.start()
    client.init_weights()

    # weight initialization
    weights = {}
    for i in range(2):
        weights["w" + str(i)] = tensor.random((3, 3))

    client.weights = weights
    client.push()
    max_epoch = 3
    for i in range(max_epoch):
        print(f"On epoch {i}:")

        # Pull from Server
        client.pull()
        print(client.weights)

        # Update locally
        for k, v in client.weights.items():
            client.weights[k] += global_rank + 1

        # Push to Server
        client.push()
        print(client.weights)

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_rank", default=0, type=int)
    args = parser.parse_args()

    test(args.global_rank)
