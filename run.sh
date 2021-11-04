#!/bin/bash

set -e

NUM_CLIENTS=10
LOG_PATH=$(date +%Y%m%d_%H%M%S)

python -m src.server.main -m 2 &

mkdir -p log/$LOG_PATH

for ((i = 0; i < NUM_CLIENTS; i++)); do
  sleep 1
  python -m src.client.main --model cnn --data mnist -m 2 -i $i -d non-iid >log/$LOG_PATH/log_$i.txt &
done

exit
