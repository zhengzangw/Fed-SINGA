#!/usr/bin/env python3
import argparse
import os

server_command = {
    "test": "python -m tests.helpers.start_app --mode server --num_client",
    "train": "python -m src.server.main -m",
}
client_command = {
    "test": "python -m tests.helpers.start_app --mode client --global_rank",
    "train": "python -m src.client.main --model cnn --data mnist -m 10 -d non-iid -i",
}
server_fmt = """
version: "3.3"

services:
  server:
    build:
      context: ../
      dockerfile: ./dockerfile
    network_mode: host
    volumes:
      - ../:/app
    command: {} {:d}
"""
client_fmt = """
  client_{:d}:
    build:
      context: ../
      dockerfile: ./dockerfile
    network_mode: host
    depends_on:
      - server
    volumes:
      - ../:/app
    command: {} {:d}
"""


def main(args):
    output_txt = server_fmt.format(server_command[args.mode], args.client)
    for i in range(args.client):
        output_txt += client_fmt.format(i, client_command[args.mode], i)

    if args.mode == "train":
        file_name = f"docker-compose.train_{args.client}.yml"
    else:
        file_name = f"docker-compose.client_{args.client}.yml"
    dir_name = os.path.dirname(__file__)
    file_name = os.path.join(dir_name, file_name)
    with open(file_name, "w") as f:
        f.write(output_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["test", "train"], default="test")
    parser.add_argument("-c", "--client", type=int, default=1)
    args = parser.parse_args()
    main(args)
