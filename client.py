from copy import deepcopy
import warnings
from collections import OrderedDict
import utils

import flwr as fl
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Define our model as multi-layer perceptron"""
    # Dimensions are random numbers, 4 layer NN
    def __init__(self, input_dim=7, hidden_layer_dims=[10, 20, 5]):
        super(Net, self).__init__()
        layers = []
        for i in range(len(hidden_layer_dims)):
            layer = [
                nn.Linear(  # Each linear is a matrix multiplication
                    # Matrix multiplication; these 2 lines are 2 dimensions
                    hidden_layer_dims[i - 1] if i > 0 else input_dim,
                    hidden_layer_dims[i],
                )
            ]
            if i < len(hidden_layer_dims) - 1:
                # Non linearities to make the NN not linear
                # output = max(xw_(7 * 10), 0)
                # 30 dimensional vectors
                layer.append(nn.ReLU()) # ReLU - non linearity
                # softmax function to get the score

            layers.append(nn.ModuleList(layer))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        for layer in self.layers:
            for func in layer:
                # print("x shape:", x.shape) # [1, 400, 6]
                x = func(x) # 400 x 6 and 7 x 10 can't be multiplied
        return x


def train(
    net,
    trainloader,
    epochs=3,
    lr=0.003,
    momentum=0.9,
    testloader=None,
    proximal_mu=0,
):
    """Train the model on the training set."""
    global_net = deepcopy(net)
    criterion = torch.nn.CrossEntropyLoss()  # Assuming universal loss function across all clients
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()

    correct, total = 0, 0

    for _ in range(epochs):
        for x, y in tqdm(trainloader):  # make 2 different datasets so both aren't training on the same one
            #  why does tqdm bar go away
            # currently batching all together
            # .squeeze by the size of the paritition
            optimizer.zero_grad()
            pred = net(x.to(DEVICE).float())
            pred = pred[0, :, :]  # Removes dimension if the dimension is 1 - flattens list to inner dimension
            y = y[0, :]
            loss = criterion(pred, y.to(DEVICE).to(dtype=torch.uint8)) 
            # Proximal term
            # This implementation is adapted from
            # https://epione.gitlabpages.inria.fr/flhd/federated_learning/FedAvg_FedProx_MNIST_iid_and_noniid.html
            loss += proximal_mu / 2 * utils.difference_models_norm_2(net, global_net)
            loss.backward()
            optimizer.step()
            correct += (torch.max(pred.data, 1)[1] == y).sum().item()
            total += y.size(0)
        print(f"Train accuracy: {correct / total}")
        if testloader is not None: # Evaluate the code
            loss, accuracy = test(net, testloader)
            print(f"Test Loss: {loss}, Accuracy: {accuracy}")


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for x, y in tqdm(testloader):
            outputs = net(x.to(DEVICE).float())
            outputs = torch.squeeze(outputs)
            # print(outputs.shape) # 25 x 400
            # 312 x 6 and 7 x 10 can't be multiplied

            labels = torch.squeeze(y.to(DEVICE), dim=0).to(dtype=torch.uint8)
            # print("labels shape:", labels.shape) # 25 x 312
            # access  all the tensors of 1
            # dimension and get only 7 values
            # in that dimension
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.training_data, self.testing_data, self.val_data = utils.load_partition(
            args.partition, args.num_clients, args.data_path, batch_size=args.batch_size
        )
        self.args = args
        self.net = Net()

    # Config is never used in the below functions but is expected by the
    # numpy client
    def get_parameters(self, config={}):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config={}):
        self.set_parameters(parameters)
        train(
            self.net,
            self.training_data,
            epochs=self.args.local_epochs,
            lr=self.args.lr,
            momentum=self.args.momentum,
            proximal_mu=self.args.proximal_mu,
        )  # Adjust number of local updates b/t communication rounds
        return self.get_parameters(config={}), len(self.training_data.dataset), {}

    def evaluate(self, parameters, config={}):
        # assert self.args.eval_dataset == 'test' or self.args.eval_dataset == 'val'
        dataset = self.testing_data # if self.args.eval_dataset == 'test' else self.val_data
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, dataset)
        print(loss, accuracy)
        return loss, len(dataset), {"accuracy": accuracy}


def run_baseline(args):
    training_data, testing_data = utils.load_partition(
        0, 1, args.data_path, batch_size=args.batch_size
    )

    net = Net()
    train(
        net,
        training_data,
        epochs=args.local_epochs,
        testloader=testing_data, # Evaluates performance every every local epoch
    )
    loss, accuracy = test(net, testing_data)
    print(f"Loss: {loss}, Accuracy: {accuracy}")


def main():
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--baseline",
        action='store_true',
        help="Whether to run a baseline experiment where a single client \
        trains on the entire dataset (i.e., a non-federated approach)"
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        required=False,
        help="Specifies the artificial data partition to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=12,
        required=False,
        help="The number of clients to use",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        required=False,
        help="Batch size to use on each client for training",
    )
    parser.add_argument(
        "-lr",
        type=float,
        default=0.0008,
        required=False,
        help="Learning rate",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
        required=False,
        help="Number of local epochs",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.87,
        required=False,
        help="Momentum for SGD with momentum",
    )
    # Arguments that deal exclusively with data
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data",
    )
    parser.add_argument(
        "--proximal-mu",
        type=float,
        required=False,
        default=0,
        help="The mu for the proximal term; if this is non-zero, this adds a proximal \
              term to the loss as proposed in the FedProx paper. If this is 0, no proximal \
              term is added."
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        required=False,
        default="val",
        help="Whether to evaluate on the test or the val set. Options are test or val"
    )

    args = parser.parse_args()
    assert args.partition >= 0 and args.partition < args.num_clients, \
        f"Partition {args.partition} is not possible with {args.num_clients} clients."

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    if args.baseline:
        run_baseline(args)
    else:
        client = FlowerClient(args)
        fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


if __name__ == "__main__":
    main()
