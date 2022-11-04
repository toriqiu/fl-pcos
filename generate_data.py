import argparse
import os
import random

import numpy as np
import pandas as pd

BIRTH_CONTROLS = list(range(5))
HORMONES = ['LH-FSH', 'Testosterone', 'DHEA-S', 'Prolactin', 'ANDRO', 'Estradiol', 'AMH']

# ######################################################################################
# Model correlations for BCs: LH-FSH ratio, Total Testosterone, DHEA-S,
# Prolactin, Androstenedione, Estradiol, Anti-Mullerian (AMH)
#
# *[NC] means no correlation, so model data will be randomly assigned to diagnostic range
# ######################################################################################

BC_RANGES = {
    0: {'LH-FSH': (2, 2.5), 'Testosterone': (121, 130.9), 'DHEA-S': (200, 300.9), 'Prolactin': (25, 40),
        'ANDRO': (1.1, 1.5), 'Estradiol': (60, 120), 'AMH': (8.1, 10)},

    1: {'LH-FSH': (3.1, 3.5), 'Testosterone': (86, 100.9), 'DHEA-S': (301, 350.9), "Prolactin": (31, 35.9),
        "ANDRO": (1.6, 2.0), "Estradiol": (81, 100.9), "AMH": (5, 10)},

    2: {'LH-FSH': (2, 3.5), "Testosterone": (101, 110.9), "DHEA-S": (351, 400.9), "Prolactin": (25, 40),
        "ANDRO": (0.4, 0.7), "Estradiol": (101, 120), "AMH": (5, 10)},

    3: {'LH-FSH': (2.6, 3), "Testosterone": (131, 150), "DHEA-S": (200, 430), "Prolactin": (36, 40),
        "ANDRO": (0.8, 1.0), "Estradiol": (60, 80.9), "AMH": (5, 6.5)},

    4: {'LH-FSH': (2, 3.5), "Testosterone": (111, 120.9), "DHEA-S": (401, 430), "Prolactin": (25, 30.9),
        "ANDRO": (2.1, 2.7), "Estradiol": (60, 120), "AMH": (6.6, 8)}
}

COMPLETE_RANGES = {'LH-FSH': (2, 3.5), "Testosterone": (86, 150), "DHEA-S": (200, 430), "Prolactin": (25, 40),
          "ANDRO": (0.4, 2.7), "Estradiol": (60, 120), "AMH": (5, 10)}


def _generate_data_for_client(
    args,
    num_points,
    data_dir,
    split,
    probabilities=np.array([1] * len(BIRTH_CONTROLS)) / len(BIRTH_CONTROLS)
):
    filepath = os.path.join(data_dir, f"{split}.csv")

    bc_counts = np.random.multinomial(num_points, probabilities)

    df_dict = {hormone: [] for hormone in HORMONES}
    df_dict['BC'] = []

    for bc, bc_count in enumerate(bc_counts):
        for _ in range(bc_count):
            for hormone in BC_RANGES[bc]:
                hormone_range = BC_RANGES[bc][hormone]
                lower, upper = hormone_range

                # Normalize to between 0 and 1
                lower = (lower - COMPLETE_RANGES[hormone][0]) / (COMPLETE_RANGES[hormone][1] - COMPLETE_RANGES[hormone][0])
                upper = (upper - COMPLETE_RANGES[hormone][0]) / (COMPLETE_RANGES[hormone][1] - COMPLETE_RANGES[hormone][0])

                # Center
                lower -= 0.5 # Center
                upper -= 0.5 # Center

                df_dict[hormone].append(random.uniform(lower, upper))
            if np.random.random() < args.error_prob:
                # Sample a different birth control but leave the hormones the same
                bc = np.random.choice(5)
            df_dict['BC'].append(bc)

    df = pd.DataFrame(data=df_dict)
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle
    df.to_csv(filepath, index=False)


def generate_data(args):
    filepath = \
        f"""data/clients_{args.num_clients}_iid_{args.iid}_error_prob_{args.error_prob}_sizes_\
{args.min_dataset_size}to{args.max_dataset_size}/"""
    os.makedirs(filepath, exist_ok=True)
    print(filepath)

    for client in range(args.num_clients):
        # If min and max size are the same, train_set_size is always args.min_dataset_size
        train_set_size = np.random.randint(args.min_dataset_size, args.max_dataset_size + 1)
        test_set_size, val_set_size = train_set_size // 4, train_set_size // 4

        if args.iid:
            probabilities = np.array([1] * len(BIRTH_CONTROLS)) / len(BIRTH_CONTROLS)
        else:
            probabilities = np.random.dirichlet([1] * len(BIRTH_CONTROLS))

        client_data_dir = os.path.join(filepath, str(client))
        
        os.makedirs(client_data_dir, exist_ok=True)

        _generate_data_for_client(
            args, train_set_size, client_data_dir, "train", probabilities=probabilities
        )
        _generate_data_for_client(
            args, test_set_size, client_data_dir, "test", probabilities=probabilities
        )
        _generate_data_for_client(
            args, val_set_size, client_data_dir, "val", probabilities=probabilities
        )


# Generate IID vs. non-IID data using PDFs - can be a flag in the command line
def main():
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--num-clients",
        type=int,
        default=12,
        required=False,
        help="The number of clients to use",
    )
    parser.add_argument(
        "--iid",
        action="store_true",
        help="Whether or not to generate iid data",
    )
    parser.add_argument(
        "--error-prob",
        type=float,
        required=False,
        default=0,
        help="The probability of choosing another drug for a patient \
        with given hormone characteristics.",
    )
    parser.add_argument(
        "--min-dataset-size",
        type=int,
        required=False,
        default=12_500,
        help="The minimum size of a client train dataset. If this is the same as --max-dataset-size, all clients \
        will have the same dataset size. Test and val set sizes are automatically chosen as 25% of each client's train set size.",
    )
    parser.add_argument(
        "--max-dataset-size",
        type=int,
        required=False,
        default=12_500,
        help="The maximum size of a client train dataset. If this is the same as --min-dataset-size, all clients \
        will have the same dataset size. Test and val set sizes are automatically chosen as 25% of each client's train set size.",
    )

    args = parser.parse_args()
    generate_data(args)

if __name__ == "__main__":
    main()
