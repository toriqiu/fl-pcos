import os

import torch
import random
import numpy
from torch.utils.data import DataLoader
import warnings
from torch.utils.data import Dataset


warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create a custom Dataset type for DataLoader
class MyData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        info = torch.from_numpy(self.df[:, :-1])  # Hormonal levels
        label = torch.from_numpy(self.df[:, -1])  # BC
        return info, label


def load_data(partition, num_partitions, data_path):
    """Load training and test set."""
    assert len(os.listdir(data_path)) == num_partitions, \
        f"Data path {data_path} contains {len(os.listdir(data_path))} datasets but there are {num_partitions} partitions."

    train_data_path = os.path.join(data_path, str(partition), "train.csv")
    test_data_path = os.path.join(data_path, str(partition), "test.csv")
    val_data_path = os.path.join(data_path, str(partition), "val.csv")

    training_data = numpy.loadtxt(train_data_path, delimiter=',', skiprows=1)
    testing_data = numpy.loadtxt(test_data_path, delimiter=',', skiprows=1)
    val_data = numpy.loadtxt(val_data_path, delimiter=',', skiprows=1)

    num_examples = {
        "trainset": len(training_data), "testset": len(testing_data), "valset": len(val_data)
    }

    return training_data, testing_data, val_data, num_examples


def load_partition(partition, num_partitions, data_path, batch_size=32):
    """Load 1/10th of the training and test data to simulate a partition."""
    training_data, testing_data, val_data, num_examples = load_data(partition, num_partitions, data_path)

    n_train = int(num_examples["trainset"])
    print("n_train:", n_train)

    return DataLoader(dataset=MyData(training_data), batch_size=batch_size, shuffle=True), \
           DataLoader(dataset=MyData(testing_data)), \
           DataLoader(dataset=MyData(val_data))


def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def difference_models_norm_2(model_1, model_2):
    """
    Return the norm 2 difference between the two model parameters
    Copied from https://epione.gitlabpages.inria.fr/flhd/federated_learning/FedAvg_FedProx_MNIST_iid_and_noniid.html
    """
    
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    
    return norm
