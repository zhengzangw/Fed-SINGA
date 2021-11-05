#!/usr/bin/env python3

from ..proto.utils import parseargs
from .app import Server

if __name__ == "__main__":
    args = parseargs()

    server = Server(num_clients=args.num_clients, host=args.host, port=args.port, secure=args.secure)
    server.start()

    for i in range(args.max_epoch):
        print(f"On epoch {i}:")
        if i > 0:
            # Push to Clients
            server.push()

        # Collects from Clients
        server.pull()

    server.close()
