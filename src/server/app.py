#!/usr/bin/env python3
import socket
import struct

from google.protobuf.message import Message

from ..proto import interface_pb2 as proto
from ..proto import utils


class Server:
    """Server sends and receives protobuf messages"""

    def __init__(self, host: str = "127.0.0.1", port: str = 1234, pack_format="Q") -> None:
        """Class init method

        Args:
            host (str, optional): Host ip address. Defaults to '127.0.0.1'.
            port (str, optional): Port. Defaults to 1234.
        """
        self.host = host
        self.port = port
        self.sock = socket.socket()
        self.pack_format = pack_format
        self.conn = None

    def start(self) -> None:
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        conn, addr = self.sock.accept()
        print("Connected by", addr)
        self.conn = conn

    def recv(self, message: Message):
        buffer_size = struct.Struct(self.pack_format).size
        data_len_buffer = utils.receive_all(self.conn, buffer_size)
        (data_len,) = struct.unpack(f">{self.pack_format}", data_len_buffer)

        message.ParseFromString(utils.receive_all(self.conn, data_len))
        return message

    def send(self, data: Message) -> None:
        utils.send_all(self.conn, data, self.pack_format)

    def close(self) -> None:
        self.sock.close()


if __name__ == "__main__":
    server = Server()
    server.start()

    max_epoch = 3
    for i in range(max_epoch):
        print(f"On epoch {i}:")

        # Push weights
        data = proto.WeightsExchange()
        data.op_type = proto.PULL
        data.weights = "Global weights"
        server.send(data)

        # Pull weights
        data = proto.WeightsExchange()
        server.recv(data)
        print(data)

    server.close()
