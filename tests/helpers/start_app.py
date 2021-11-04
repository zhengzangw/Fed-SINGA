import argparse

from singa import tensor

from src.client.app import Client
from src.server.app import Server

max_epoch = 3


def main_server(s):
    s.start()

    s.pull()

    for i in range(max_epoch):
        print(f"[Server] On epoch {i}")
        s.push()
        s.pull()
    s.close()


def main_client(c):
    c.start()

    # weight initialization
    weights = {}
    for i in range(2):
        weights["w" + str(i)] = tensor.random((3, 3))
    c.weights = weights
    c.push()

    for i in range(max_epoch):
        print(f"[Client {c.global_rank}] On epoch {i}")

        # Pull from Server
        c.pull()

        # Update locally
        for k in c.weights.keys():
            c.weights[k] += c.global_rank + 1

        # Push to Server
        c.push()
    c.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "client"])
    parser.add_argument("--num_clients", default=1, type=int)
    parser.add_argument("--global_rank", default=0, type=int)
    parser.add_argument("--secure", action="store_true")
    parser.add_argument("--port", default=1234, type=int)
    args = parser.parse_args()

    if args.mode == "server":
        s = Server(num_clients=args.num_clients, port=args.port, secure=args.secure)
        main_server(s)
    elif args.mode == "client":
        c = Client(global_rank=args.global_rank, port=args.port, secure=args.secure)
        main_client(c)
