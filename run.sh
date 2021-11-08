#!/bin/bash

set -e

NUM_CLIENTS=10
NUM_EPOCH=3

LOG_PATH=$(date +%Y%m%d_%H%M%S)

python -m src.server.main -m $NUM_EPOCH -s True --num_clients $NUM_CLIENTS &

mkdir -p log/$LOG_PATH

for ((i = 0; i < NUM_CLIENTS; i++)); do
  sleep 1
  python -m src.client.main --model mlp --data mnist -m $NUM_EPOCH -i $i -d non-iid -s True >log/$LOG_PATH/log_$i.txt &
done

exit
