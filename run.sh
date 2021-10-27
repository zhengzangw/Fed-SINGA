#!/bin/bash

NUM_CLIENTS=10
NUM_JOBS="\j" # The prompt escape for number of jobs currently running
LOG_PATH=$(date +%Y%m%d_%H%M%S)

trap ctrl_c INT

cd src/server && python server.py &

cd src/client
mkdir -p log/$LOG_PATH

for ((i = 0; i < NUM_CLIENTS; i++)); do
  sleep 1
  python client.py --model cnn --data mnist -m 100 -i $i -d non-iid > log/$LOG_PATH/log_$i.txt &
done

exit

