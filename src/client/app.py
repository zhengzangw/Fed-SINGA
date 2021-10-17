#!/usr/bin/env python3
import socket
import struct

from google.protobuf.message import Message

from ..proto import interface_pb2 as proto
from ..proto import utils


class Client:
    """Client sends and receives protobuf messages"""

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

    def start(self) -> None:
        self.sock.connect((self.host, self.port))
        print(f"Connect to {self.host}:{self.port}")

    def recv(self, message: Message):
        buffer_size = struct.Struct(self.pack_format).size
        data_len_buffer = utils.receive_all(self.sock, buffer_size)
        (data_len,) = struct.unpack(f">{self.pack_format}", data_len_buffer)

        message.ParseFromString(utils.receive_all(self.sock, data_len))
        return message

    def send(self, data: Message) -> None:
        utils.send_all(self.sock, data, self.pack_format)

    def close(self) -> None:
        self.sock.close()


if __name__ == "__main__":
    client = Client()
    client.start()

    max_epoch = 3
    for i in range(max_epoch):
        print(f"On epoch {i}:")

        # Receive weights
        data = proto.WeightsExchange()
        client.recv(data)
        print(data)

        # Update locally
        data = proto.WeightsExchange()
        data.op_type = proto.PUSH
        data.weights = "Serialized local weights"
        client.send(data)

    client.close()
