#!/bin/bash

NUM_CLIENTS=10
NUM_JOBS="\j"  # The prompt escape for number of jobs currently running

trap ctrl_c INT

function ctrl_c() {
  pkill -P $$
}

for e in {201..400}
do
  echo "epoch: $e"
  for ((i=0; i<NUM_CLIENTS; i++)); do
  #  while (( $(NUM_JOBS@P) >= NUM_CLIENTS )); do
  #    wait -n
  #  done
    sleep .05
    python mycnn.py cnn mnist -m 1 -i $i >> log_$i.txt &
  done
  wait
  python aggregate.py
done
exit
