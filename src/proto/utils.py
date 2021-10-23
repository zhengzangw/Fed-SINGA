import pickle
import struct

from google.protobuf.message import Message
from singa import tensor


def receive_all(conn, size):
    buffer = b""
    while size > 0:
        chunk = conn.recv(size)
        if not chunk:
            raise RuntimeError("connection closed before chunk was read")
        buffer += chunk
        size -= len(chunk)
    return buffer


def send_int(conn, i, pack_format="I"):
    data = struct.pack(f"!{pack_format}", i)
    conn.sendall(data)


def receive_int(conn, pack_format="I"):
    buffer_size = struct.Struct(pack_format).size
    data = receive_all(conn, buffer_size)
    (data,) = struct.unpack(f"!{pack_format}", data)
    return data


def send_message(conn, data: Message, pack_format="I") -> None:
    send_int(conn, data.ByteSize(), pack_format)
    conn.sendall(data.SerializePartialToString())


def receive_message(conn, data: Message, pack_format="I"):
    data_len = receive_int(conn, pack_format)
    data.ParseFromString(receive_all(conn, data_len))
    return data


def serialize_tensor(t):
    return pickle.dumps(tensor.to_numpy(t), protocol=0)


def deserialize_tensor(t):
    return tensor.from_numpy(pickle.loads(t))
