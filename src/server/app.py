#!/usr/bin/env python3

import socket
from collections import defaultdict
from typing import Dict, List

from singa import tensor

from ..proto import interface_pb2 as proto
from ..proto import utils


class Server:
    """Server sends and receives protobuf messages.

    Create and start the server, then use pull and push to communicate with clients.

    Attributes:
        num_clients (int): Number of clients.
        host (str): Host address of the server.
        port (str): Port of the server.
        sock (socket.socket): Socket of the server.
        conns (List[socket.socket]): List of num_clients sockets.
        addrs (List[str]): List of socket address.
        weights (Dict[Any]): Weights stored on server.
    """

    def __init__(
        self,
        num_clients=1,
        host: str = "127.0.0.1",
        port: str = 1234,
    ) -> None:
        """Class init method

        Args:
            num_clients (int, optional): Number of clients in training.
            host (str, optional): Host ip address. Defaults to '127.0.0.1'.
            port (str, optional): Port. Defaults to 1234.
        """
        self.num_clients = num_clients
        self.host = host
        self.port = port

        self.sock = socket.socket()
        self.conns = [None] * num_clients
        self.addrs = [None] * num_clients

        self.weights = {}

    def __start_connection(self) -> None:
        """Start the network connection of server."""
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        print("Server started.")

    def __start_rank_pairing(self) -> None:
        """Start pair each client to a global rank"""
        for _ in range(self.num_clients):
            conn, addr = self.sock.accept()
            rank = utils.receive_int(conn)
            self.conns[rank] = conn
            self.addrs[rank] = addr
            print(f"[Server] Connected by {addr} [global_rank {rank}]")
        assert None not in self.conns

    def start(self) -> None:
        """Start the server.

        This method will first bind and listen on the designated host and port.
        Then it will connect to num_clients clients and maintain the socket.
        In this process, each client shall provide their rank number.
        """
        self.__start_connection()
        self.__start_rank_pairing()

    def close(self) -> None:
        """Close the server."""
        self.sock.close()

    def aggregate(self, weights: Dict[str, List[tensor.Tensor]]) -> Dict[str, tensor.Tensor]:
        """Aggregate collected weights to update server weight.

        Args:
            weights (Dict[str, List[tensor.Tensor]]): The collected weights.

        Returns:
            Dict[str, tensor.Tensor]: Updated weight stored in server.
        """
        for k, v in weights.items():
            self.weights[k] = sum(v) / self.num_clients
        return self.weights

    def pull(self) -> None:
        """Server pull weights from clients.

        Namely clients push weights to the server. It is the gather process.
        """
        # open space to collect weights from clients
        datas = [proto.WeightsExchange() for _ in range(self.num_clients)]
        weights = defaultdict(list)
        # receive weights sequentially
        for i in range(self.num_clients):
            utils.receive_message(self.conns[i], datas[i])
            for k, v in datas[i].weights.items():
                weights[k].append(utils.deserialize_tensor(v))
        # aggregation
        self.aggregate(weights)

    def push(self) -> None:
        """Server push weights to clients.

        Namely clients pull weights from server. It is the scatter process.
        """
        message = proto.WeightsExchange()
        message.op_type = proto.SCATTER
        for k, v in self.weights.items():
            message.weights[k] = utils.serialize_tensor(v)

        for conn in self.conns:
            utils.send_message(conn, message)
