import struct

from google.protobuf.message import Message


def receive_all(conn, size):
    buffer = b""
    while size > 0:
        chunk = conn.recv(size)
        if not chunk:
            raise RuntimeError("connection closed before chunk was read")
        buffer += chunk
        size -= len(chunk)
    return buffer


def send_all(conn, data: Message, pack_format="I") -> None:
    data_len = struct.pack(f">{pack_format}", data.ByteSize())
    conn.sendall(data_len)
    conn.sendall(data.SerializePartialToString())
