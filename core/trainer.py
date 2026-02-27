import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

class Trainer:

    def train(self, model, X_train, y_train, config):

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        model.train()

        for _ in range(config["epochs"]):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        return model