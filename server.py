import argparse
import os

import flwr as fl
import utils
import warnings
import torch
import numpy as np
from flwr.common import parameters_to_ndarrays

from client import Net

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Use when dataset sizes differ between clients
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

class FedAdamWrapper(fl.server.strategy.FedAdam):
    """
    The flwr library for some reason requires an initial_parameters argument
    just for FedOpt and related methods, so we pass this in.
    """
    def __init__(self, **kwargs):
        parameters = [param.detach() for param in list(Net().parameters())]
        parameters = fl.common.ndarrays_to_parameters(parameters)
        super().__init__(initial_parameters=parameters, **kwargs)

STRATEGY_FUNCS = {
    "FedAvg": fl.server.strategy.FedAvg,
    "FedAvgM": fl.server.strategy.FedAvgM,
    # For now, use the same strategy as FedAvg for FedProx,
    # as FedAvg and FedProx are the same on the server side
    # as long as there are no failures.
    "FedProx": fl.server.strategy.FedAvg,
    "FedAdam": FedAdamWrapper,
}

AGG_FUNCS = {
    "weighted": weighted_average,
}

def get_save_model_strategy(base_strategy, name):
    """
    Wraps the provided base strategy in code to save off checkpoints at each round.
    """
    class SaveModelStrategy(base_strategy):
        def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,
        ):
            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

                os.makedirs("checkpoints/", exist_ok=True)
                # Save aggregated_ndarrays
                print(f"Saving round {server_round} aggregated_ndarrays...")
                np.savez(f"checkpoints/{name}-round-{server_round}-weights.npz", *aggregated_ndarrays)

            return aggregated_parameters, aggregated_metrics

    return SaveModelStrategy

def start_server(
    strategy_func,
    strategy_name,
    agg_func,
    min_available_clients=2,
    fraction_fit=0.5,
    num_rounds=8,
    name="no_name"
):
    strategy = get_save_model_strategy(strategy_func, name)(
        evaluate_metrics_aggregation_fn=agg_func,
        min_available_clients=min_available_clients,
        fraction_fit=fraction_fit,   # Fraction of clients which should participate in each round
    )

    if strategy_name == "FedAvgM":
        print("Setting server momentum for FedAvgM")
        strategy.server_momentum = 0.8 # Set momentum on the server

    # Start Flower server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

def main():
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument(
        "--fraction-fit",
        type=float,
        required=False,
        default=0.5,
        help="The fraction of clients to sample in each communication training \
        round. Most be a float between 0 and 1.",
    )
    parser.add_argument(
        "--min-available-clients",
        type=int,
        required=False,
        default=2,
        help="The minimum number of clients that must have started up by the time \
        we start training.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        required=False,
        default=8,
        help="The number of communication rounds",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=False,
        default="FedAvg",
        help="Which strategy to use. Must be one of FedAvg or FedAvgM",
    )
    parser.add_argument(
        "--agg",
        type=str,
        required=False,
        default="weighted",
        help="Whether to use weighted or unweighted aggregation",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="no_name",
        help="Name for the run; used only for saving off checkpoints"
    )

    args = parser.parse_args()

    if args.strategy not in STRATEGY_FUNCS:
        raise NotImplementedError(f"Attempted to use a non-existent strategy f{args.strategy}")

    if args.agg not in AGG_FUNCS:
        raise NotImplementedError(f"Attempted to use a non-existent agg f{args.agg}")

    start_server(
        STRATEGY_FUNCS[args.strategy],
        args.strategy,
        AGG_FUNCS[args.agg],
        args.min_available_clients,
        args.fraction_fit,
        args.num_rounds,
        args.name,
    )

if __name__ == "__main__":
    main()
