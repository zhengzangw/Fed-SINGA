#!/usr/bin/env python3
import argparse

server_fmt = """
version: "3.3"

services:
  server:
    build:
      context: ./
      dockerfile: ./src/dockerfile
    network_mode: host
    volumes:
      - ./:/app
    command: python -m src.server.app --num_client {:d}
"""
client_fmt = """
  client_{:d}:
    build:
      context: ./
      dockerfile: ./src/dockerfile
    network_mode: host
    depends_on:
      - server
    volumes:
      - ./:/app
    command: python -m src.client.app --global_rank {:d}
"""


def main(args):
    output_txt = server_fmt.format(args.client)
    for i in range(args.client):
        output_txt += client_fmt.format(i, i)

    file_name = f"docker-compose.client_{args.client}.yml"
    with open(file_name, "w") as f:
        f.write(output_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--client", type=int, default=1)
    args = parser.parse_args()
    main(args)
