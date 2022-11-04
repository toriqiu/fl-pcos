# pcos-fl using PyTorch & Flower

## Project Setup

Start by cloning the project:

```shell
git clone https://github.com/toriqiu/pcos-fl.git
```

This will create a new directory called `pcos-fl` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- README.md
-- utils.py
```
## Environment
To create the proper environment for running things, first install miniconda. Then, run the following commands:
```shell
conda env create --file environment.yml
conda activate pricomp
```

## Run Federated Learning with PyTorch and Flower

To run all clients at once:
```shell
./run.sh
```

To run a baseline experiment where a single neural network is trained on all available traing data, run;
```shell
./baseline.sh
```
