#!/usr/bin/env python3
import socket
import struct

from google.protobuf.message import Message

from ..proto.interface_pb2 import WeightPush


class Client:
    """Client sends and receives protobuf messages"""

    def __init__(self, host: str = "127.0.0.1", port: str = 1234, pack_format="I") -> None:
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

    def send(self, data: Message) -> None:
        data_len = struct.pack(f">{self.pack_format}", data.ByteSize())
        self.sock.sendall(data_len)
        self.sock.sendall(data.SerializePartialToString())

    def close(self) -> None:
        self.sock.close()


if __name__ == "__main__":
    data = WeightPush()
    data.message = "hello"

    client = Client()
    client.start()
    client.send(data)
    client.close()
