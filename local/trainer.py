import torch
from copy import deepcopy
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console

console = Console()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float,
        device: str,
    ):
        """
        Trainer class for training the model

        Args:
            model: nn.Module
                The model to be trained
            dataloader: DataLoader
                The traning dataloader for the model
            device: str
                The device to be used for training
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.device = device

    def train(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_sum = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_sum / len(self.train_loader)

    def train_epochs(self, n_epoch):
        """
        Local training of one client for one round
        """
        for _ in range(n_epoch):
            self.train()

    def test(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        loss, acc = 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss /= len(self.test_loader)
        acc /= len(self.test_loader.dataset)
        return loss, acc
