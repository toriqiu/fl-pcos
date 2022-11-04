#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Generates 12 iid datasets, with no error (replicates what we have)
# python generate_data.py --num-clients 12 --iid
# Generates 12 iid datasets, each with an error prob of 0.1
# python generate_data.py --num-clients 12 --iid --error-prob 0.1
# Generates 12 non-iid datasets with error prob of 0.1, each of different length between 200 and 20000
#python generate_data.py --num-clients 12 --error-prob 0.1 --min-dataset-size 200 --max-dataset-size 20000
python generate_data.py --num-clients 12 --iid --error-prob 0.1 --min-dataset-size 200 --max-dataset-size 20000

# Generates 12 non-iid datasets with error prob of 0.1, each of same lengths
python generate_data.py --num-clients 12 --error-prob 0.1 --min-dataset-size 12500 --max-dataset-size 12500
# python generate_data.py --num-clients 12 --error-prob 0.1 --min-dataset-size 200 --max-dataset-size 20000
# Generates 12 non-iid datasets with error prob of 0.1, each of the same length
# python generate_data.py --num-clients 12 --error-prob 0.1

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait