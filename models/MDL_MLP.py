import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

BATCH_SIZE = 64
EPOCHS = 6

class MDL_MLP:
    def __init__(self):
        pass

    def run(self, state, action):
        torch.manual_seed(123)
        data_x, data_y = state

        train_x, train_y, test_x, test_y = (
            torch.from_numpy(data_x[action == 1]).float().cuda(),
            torch.from_numpy(data_y[action == 1]).long().cuda(),
            torch.from_numpy(data_x[action == 0]).float().cuda(),
            torch.from_numpy(data_y[action == 0]).long().cuda()
        )

        model = nn.Sequential(
                    nn.Linear(data_x.shape[1], 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2)
                ).cuda()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_ds = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        best_auc = 0.0
        best_y_pred = np.zeros(len(test_y))
        for epoch in range(EPOCHS):
            for _, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target[:,1])
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                y_pred_prob = nn.Softmax(dim=1)(model(test_x)).cpu().numpy()
                auc = roc_auc_score(data_y[action == 0], y_pred_prob)
                if auc > best_auc:
                    best_auc = auc
                    best_y_pred = y_pred_prob

        return best_y_pred
