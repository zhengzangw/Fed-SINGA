#!/usr/bin/env python3
import argparse
import socket
import sys
from singa import tensor

sys.path.append("..")
from proto import interface_pb2 as proto
from proto import utils

from app import Server


def test(num_clients=1):
    server = Server(num_clients=num_clients)
    server.start()
    server.init_weights()

    max_epoch = 3
    for i in range(max_epoch):
        print(f"On epoch {i}:")

        # Push to Clients
        server.push()
        print(server.weights)

        # Collects from Clients
        server.pull()
        print(server.weights)

    server.close()



server = Server(num_clients=10)
server.start()
server.init_weights()

max_epoch = 50
for i in range(max_epoch):
    print(f"On epoch {i}:")
    if i > 0 :
        # Push to Clients
        server.push()

    # Collects from Clients
    server.pull()

server.close()

