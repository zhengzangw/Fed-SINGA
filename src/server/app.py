#!/usr/bin/env python3

import argparse
import socket

from ..proto import interface_pb2 as proto
from ..proto import utils


class Server:
    """Server sends and receives protobuf messages"""

    def __init__(
        self, host: str = "127.0.0.1", port: str = 1234, pack_format: str = "Q", num_clients=1
    ) -> None:
        """Class init method

        Args:
            host (str, optional): Host ip address. Defaults to '127.0.0.1'.
            port (str, optional): Port. Defaults to 1234.
        """
        self.host = host
        self.port = port
        self.sock = socket.socket()

        self.num_clients = num_clients
        self.conns = [None] * num_clients
        self.addrs = [None] * num_clients

        self.pack_format = pack_format

    def init_weights(self):
        self.weights = {}

    def start(self) -> None:
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        print("Server started.")
        for _ in range(self.num_clients):
            conn, addr = self.sock.accept()
            rank = utils.receive_int(conn)
            self.conns[rank] = conn
            self.addrs[rank] = addr
            print(f"[Server] Connected by {addr} [global_rank {rank}]")

    def pull(self):
        datas = [proto.WeightsExchange() for _ in range(self.num_clients)]
        for i in range(self.num_clients):
            utils.receive_message(self.conns[i], datas[i], self.pack_format)

        weights = {}
        for k in datas[0].weights.keys():
            weights[k] = []
        for i in range(self.num_clients):
            for k, v in datas[i].weights.items():
                weights[k].append(utils.deserialize_tensor(v))

        for k, v in weights.items():
            self.weights[k] = sum(v) / self.num_clients

    def push(self) -> None:
        message = proto.WeightsExchange()
        message.op_type = proto.PULL
        for k, v in self.weights.items():
            message.weights[k] = utils.serialize_tensor(v)

        for conn in self.conns:
            utils.send_message(conn, message, self.pack_format)

    def close(self) -> None:
        self.sock.close()


def test(num_clients=1):
    server = Server(num_clients=num_clients)
    server.start()
    server.init_weights()

    # weight initialization
    server.pull()

    max_epoch = 3
    for i in range(max_epoch):
        print(f"On epoch {i}:")

        # Push to Clients
        server.push()
        print(server.weights)

        # Collects from Clients
        server.pull()
        print(server.weights)

    server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", default=1, type=int)
    args = parser.parse_args()

    test(args.num_clients)
