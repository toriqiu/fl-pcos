#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

python server.py --fraction-fit 0.5 --min-available-clients 2 --num-rounds 8 --strategy FedAdam&
sleep 15  # Sleep for 3s to give the server enough time to start

for i in `seq 0 11`; do
    echo "Starting client $i"
    # Non-iid, different dataset sizes
    python client.py --partition=${i} --num-clients=12 --data-path data/clients_12_iid_False_error_prob_0.1_sizes_200to20000 --eval-dataset test&
    # iid, same dataset sizes
    # python client.py --partition=${i} --num-clients=12 --data-path data/clients_12_iid_True_error_prob_0.1_sizes_12500to12500 --eval-dataset test&
    # Non-iid, same dataset sizes
    # python client.py --partition=${i} --num-clients=12 --data-path data/clients_12_iid_False_error_prob_0.1_sizes_12500to12500 --eval-dataset test&
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
