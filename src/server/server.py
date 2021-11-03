#!/usr/bin/env python3

from .app import Server

server = Server(num_clients=10)
server.start()
server.init_weights()


max_epoch = 50
for i in range(max_epoch):
    print(f"On epoch {i}:")
    if i > 0:
        # Push to Clients
        server.push()

    # Collects from Clients
    server.pull()

server.close()
