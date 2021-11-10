
---

<div align="center">

# Fed-SINGA

[![SINGA](https://img.shields.io/badge/SINGA-803300?logoColor=white)](https://singa.apache.org/)
<!-- ![coverage](https://img.shields.io/badge/coverage-25%25-yellowgreen) -->
![license](https://img.shields.io/badge/license-Apache-green)

</div>

## Description

A Federated Learning Framework using Apache SINGA.

This framework provides a codebase for developing federated learning algorithm with Apache SINGA Deep Learning Framework. Fed-SINGA simulates the real world setting by a server-client structure, where each can be placed in a docker container.

The baseline method implements Fed-Avg algorithm and Secure Aggregation.

## How to run

Install dependencies

```bash
pip install -r requirements.txt
```

Prepare the data

```bash
# mnist
python -m src.client.data.download_mnist

# bank
# 1. download the data manually from https://archive.ics.uci.edu/ml/datasets/bank+marketing
# 2. put it under the /data folder
# 3. run the following command which:
#    (1) splits the dataset into N subsets
#    (2) splits each subsets into train set and test set (8:2)
python -m src.client.data.bank N
```

Training the model

```bash
# Server and clients in host machine
bash run.sh

# Server and clients in docker containers
docker-compose -f scripts/docker-compose.train_10.yml up
```

Generate the docker training scripts

```bash
python scripts/compose_setup.py -c 10 --mode train
```

## Unit Test

Unit tests and smoke tests are provided to ensure the correctness of the implementation. Run following commands to check.

```bash
pytest --cov=src tests/
```

## Contribution

See [How-to-Contribute](contributing.md)
