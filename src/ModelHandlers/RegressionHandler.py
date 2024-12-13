import torch
from sklearn.metrics import mean_squared_error

from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler


def int_to_onehot(idx):
    one_hot = torch.zeros(2).float()
    one_hot[idx] = 1.0
    return one_hot


class Dataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class RegressionHandler(BasicHandler):
    def evaluate_model(self, loader) -> float:
        """
        Evaluate the model on validation or test data.

        Args:
            loader (DataLoader): DataLoader for validation or test set.

        Returns:
            float: Mean squared error of the model on the provided dataset.
        """
        self.model.eval()
        device = StaticConf.get_instance().conf_values.device
        self.model.to(device)

        all_preds = []
        all_targets = []

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = self.model(x_batch).detach().cpu().numpy().reshape(-1)
            all_preds.extend(preds)
            all_targets.extend(y_batch.cpu().numpy())

        mse = mean_squared_error(all_targets, all_preds)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    def train_model(self, train_loader):
        """
        Train the regression model using the provided training data loader.

        Args:
            train_loader (DataLoader): DataLoader for training.
        """
        device = StaticConf.get_instance().conf_values.device
        self.model.to(device)
        self.model.train()

        for epoch in range(StaticConf.get_instance().conf_values.num_epoch):
            running_loss = 0.0

            for curr_x, curr_y in train_loader:
                curr_x, curr_y = curr_x.to(device), curr_y.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(curr_x.float())
                loss = self.loss_func(outputs, curr_y.float())

                loss.backward()
                self.optimizer.step()

                running_loss += loss

            print(f"Epoch {epoch + 1}: Loss = {running_loss:.4f}")
