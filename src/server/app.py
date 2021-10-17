#!/usr/bin/env python3
import socket
import struct

from google.protobuf.message import Message

from ..proto.interface_pb2 import WeightPush


def receive_all(conn, size):
    buffer = b""
    while size > 0:
        chunk = conn.recv(size)
        if not chunk:
            raise RuntimeError("connection closed before chunk was read")
        buffer += chunk
        size -= len(chunk)
    return buffer


class Server:
    """Server sends and receives protobuf messages"""

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
        self.sock.bind((self.host, self.port))
        self.sock.listen()

    def recv(self, message: Message):
        conn, addr = self.sock.accept()
        print("Connected by", addr)

        buffer_size = struct.Struct(self.pack_format).size
        data_len_buffer = receive_all(conn, buffer_size)
        (data_len,) = struct.unpack(f">{self.pack_format}", data_len_buffer)

        message.ParseFromString(receive_all(conn, data_len))
        return message

    def send(self, data: Message) -> None:
        data_len = struct.pack(f">{self.pack_format}", data.ByteSize())
        self.sock.sendall(data_len)
        self.sock.sendall(data.SerializePartialToString())

    def close(self) -> None:
        self.sock.close()


if __name__ == "__main__":
    server = Server()
    server.start()
    data = WeightPush()
    server.recv(data)
    print(data)
    server.close()
