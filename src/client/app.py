#!/usr/bin/env python3

import random
import socket

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from google.protobuf import message
from singa import tensor

from ..proto import interface_pb2 as proto
from ..proto import utils


class Client:
    """Client sends and receives protobuf messages.

    Create and start the server, then use pull and push to communicate with the server.

    Attributes:
        global_rank (int): The rank in training process.
        host (str): Host address of the server.
        port (str): Port of the server.
        sock (socket.socket): Socket of the client.
        weights (Dict[Any]): Weights stored locally.
    """

    def __init__(
        self,
        global_rank: int = 0,
        host: str = "127.0.0.1",
        port: str = 1234,
        secure: bool = False,
    ) -> None:
        """Class init method

        Args:
            global_rank (int, optional): The rank in training process. Defaults to 0.
            host (str, optional): Host ip address. Defaults to '127.0.0.1'.
            port (str, optional): Port. Defaults to 1234.
            secure (bool, optional): Whether use secure aggregation. Defaults to False.
        """
        self.host = host
        self.port = port
        self.global_rank = global_rank
        self.secure = secure

        self.sock = socket.socket()

        self.weights = {}

    def __start_connection(self) -> None:
        """Start the network connection to server."""
        self.sock.connect((self.host, self.port))

    def __start_rank_pairing(self) -> None:
        """Sending global rank to server"""
        utils.send_int(self.sock, self.global_rank)

    def __init_secure_aggregation(self) -> None:
        # push public key to server
        self.private_key = ec.generate_private_key(ec.SECP384R1())
        self.public_key = self.private_key.public_key()
        message = proto.WeightsExchange()
        message.op_type = proto.GATHER
        message.weights["pub"] = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        utils.send_message(self.sock, message)

        # receive number of clients
        self.num_clients = utils.receive_int(self.sock)
        self.public_keys = [None] * self.num_clients
        self.shared_keys = [None] * self.num_clients
        self.rand_g = [random.Random() for _ in range(self.num_clients)]

        # receive all public key
        message = proto.WeightsExchange()
        message = utils.receive_message(self.sock, message)
        for i in range(self.num_clients):
            self.public_keys[i] = serialization.load_pem_public_key(message.weights[str(i)])
            self.shared_keys[i] = self.private_key.exchange(ec.ECDH(), self.public_keys[i])

        # set random generator state
        for i in range(self.num_clients):
            if i != self.global_rank:
                self.rand_g[i].seed(self.shared_keys[i])

    def start(self) -> None:
        """Start the client.

        This method will first connect to the server. Then global rank is sent to the server.
        """
        self.__start_connection()
        self.__start_rank_pairing()
        if self.secure:
            self.__init_secure_aggregation()

        print(f"[Client {self.global_rank}] Connect to {self.host}:{self.port}")

    def close(self) -> None:
        """Close the server."""
        self.sock.close()

    def pull(self) -> None:
        """Client pull weights from server.

        Namely server push weights from clients.
        """
        message = proto.WeightsExchange()
        message = utils.receive_message(self.sock, message)
        for k, v in message.weights.items():
            self.weights[k] = utils.deserialize_tensor(v)

    def secure_weights(self) -> None:
        """Secure aggregation.

        Add shared random number to weights.
        """
        noise = 0
        for i in range(self.num_clients):
            if i != self.global_rank:
                sign = 1 if i < self.global_rank else -1
                n = self.rand_g[i].uniform(0, 1) * sign
                noise += n
        for k in self.weights.keys():
            self.weights[k] += noise

    def push(self) -> None:
        """Client push weights to server.

        Namely server pull weights from clients.
        """
        message = proto.WeightsExchange()
        message.op_type = proto.GATHER
        if self.secure:
            self.secure_weights()
        for k, v in self.weights.items():
            message.weights[k] = utils.serialize_tensor(v)
        utils.send_message(self.sock, message)
