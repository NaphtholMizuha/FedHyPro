import torch.utils.data
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import os


class DataSplitter:
    def __init__(
        self, dataset: Dataset, n_client, batch_size, num_workers, path, method
    ):
        self.dataset = dataset
        self.method = method
        self.n_class = 10
        self.n_client = n_client
        self.batch_size = batch_size
        self.num_workers = num_workers
        print(f"path: {path}")
        if os.path.exists(path):
            print(f"Loading partition from {path}")
            partition_dict = np.load(path, allow_pickle=True).item()
        else:
            if method == "iid":
                partition_dict = self.iid_partition()
            elif method == "dir0.5":
                partition_dict = self.dirichlet_partition(0.5)
            elif method == "dir1":
                partition_dict = self.dirichlet_partition(1)
            elif method == "pathetic":
                partition_dict = self.pathetic_partition()
            np.save(path, partition_dict)
        self.d_sizes = [len(partition_dict[i]) for i in range(n_client)]
        self.dataloaders = self.split(partition_dict)

    def iid_partition(self):
        indices = np.array([i for i in range(len(self.dataset))])
        np.random.shuffle(indices)
        indices = np.array_split(indices, self.n_client)

        return {i: indices[i] for i in range(self.n_client)}

    def pathetic_partition(self):
        partition_dict = {i: [] for i in range(self.n_client)}
        for i, (_, y) in enumerate(self.dataset):
            partition_dict[y % self.n_client].append(i)
        return partition_dict

    def dirichlet_partition(self, alpha):
        self.labels = np.array([item[1] for item in self.dataset])
        label_dist = np.random.dirichlet([alpha] * self.n_client, self.n_class)
        class_idcs = [
            np.argwhere(self.labels == y).flatten() for y in range(self.n_class)
        ]
        client_idcs = [[] for _ in range(self.n_client)]

        for k_idcs, fracs in zip(class_idcs, label_dist):
            for i, idcs in enumerate(
                np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))
            ):
                client_idcs[i] += [idcs]

        return {i: np.concatenate(idcs) for i, idcs in enumerate(client_idcs)}

    def split(self, partition_dict):
        subsets = {}
        for i, idcs in partition_dict.items():
            subsets[i] = Subset(self.dataset, idcs)

        return {
            i: DataLoader(
                subsets[i],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )
            for i in range(self.n_client)
        }

    def show_distribution(self, partition_dict: dict, path):
        mat = np.zeros([self.n_client, self.n_class])
        for client, indices in partition_dict.items():
            for idx in indices:
                mat[client][int(self.dataset.targets[idx])] += 1
        plt.matshow(mat, vmin=0, vmax=4000, cmap=plt.cm.Blues)
        plt.xlabel("Client")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel("Class")
        plt.title("Distribution")
        plt.colorbar()
        plt.savefig(path)
        plt.close()
