import pickle
import socket
import struct

from google.protobuf.message import Message
from singa import tensor


def receive_all(conn: socket.socket, size: int) -> bytes:
    """Receive a given length of bytes from socket.

    Args:
        conn (socket.socket): Socket connection.
        size (int): Length of bytes to receive.

    Raises:
        RuntimeError: If connection closed before chunk was read, it will raise an error.

    Returns:
        bytes: Received bytes.
    """
    buffer = b""
    while size > 0:
        chunk = conn.recv(size)
        if not chunk:
            raise RuntimeError("connection closed before chunk was read")
        buffer += chunk
        size -= len(chunk)
    return buffer


def send_int(conn: socket.socket, i: int, pack_format: str = "Q") -> None:
    """Send an integer from socket.

    Args:
        conn (socket.socket): Socket connection.
        i (int): Integer to send.
        pack_format (str, optional): Pack format. Defaults to "Q", which means unsigned long long.
    """
    data = struct.pack(f"!{pack_format}", i)
    conn.sendall(data)


def receive_int(conn: socket.socket, pack_format: str = "Q") -> int:
    """Receive an integer from socket.

    Args:
        conn (socket.socket): Socket connection.
        pack_format (str, optional): Pack format. Defaults to "Q", which means unsigned long long.

    Returns:
        int: Received integer.
    """
    buffer_size = struct.Struct(pack_format).size
    data = receive_all(conn, buffer_size)
    (data,) = struct.unpack(f"!{pack_format}", data)
    return data


def send_message(conn: socket.socket, data: Message, pack_format: str = "Q") -> None:
    """Send protobuf message from socket. First the length of protobuf message will be sent. Then the message is sent.

    Args:
        conn (socket.socket): Socket connection.
        data (Message): Protobuf message to send.
        pack_format (str, optional): Length of protobuf message pack format. Defaults to "Q", which means unsigned long long.
    """
    send_int(conn, data.ByteSize(), pack_format)
    conn.sendall(data.SerializePartialToString())


def receive_message(conn: socket.socket, data: Message, pack_format: str = "Q") -> Message:
    """Receive protobuf message from socket

    Args:
        conn (socket.socket): Socket connection.
        data (Message): Placehold for protobuf message.
        pack_format (str, optional): Length of protobuf message pack format. Defaults to "Q", which means unsigned long long.

    Returns:
        Message: The protobuf message.
    """
    data_len = receive_int(conn, pack_format)
    data.ParseFromString(receive_all(conn, data_len))
    return data


def serialize_tensor(t: tensor.Tensor) -> bytes:
    """Serialize a singa tensor to bytes.

    Args:
        t (tensor.Tensor): The singa tensor.

    Returns:
        bytes: The serialized tensor.
    """
    return pickle.dumps(tensor.to_numpy(t), protocol=0)


def deserialize_tensor(t: bytes) -> tensor.Tensor:
    """Recover singa tensor from bytes.

    Args:
        t (bytes): The serialized tensor.

    Returns:
        tensor.Tensor: The singa tensor.
    """
    return tensor.from_numpy(pickle.loads(t))
