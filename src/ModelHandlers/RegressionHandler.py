import copy

import torch
from sklearn.metrics import mean_squared_error
import numpy as np

from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler


# TODO: Consider data normalization and augmentation via torchvision.transforms
class Dataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x
        self.y = torch.tensor(y, dtype=torch.float32) if isinstance(y, np.ndarray) else y

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

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = self.model(x_batch).detach().view(-1).tolist()
                all_preds.extend(preds)
                all_targets.extend(y_batch.detach().view(-1).tolist())

        mse = mean_squared_error(all_targets, all_preds)
        print(f"Mean Squared Error: {mse:.3f}")
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

        best_loss = np.inf
        best_state_dict = None
        epochs_not_improved = 0
        MAX_EPOCHS_PATIENCE = 10  # TODO: Consider updating / dynamically changing

        # Dynamic Learning Rate Scheduling  # TODO: New addition, assess with and without
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        for epoch in range(StaticConf.get_instance().conf_values.num_epochs):
            running_loss = 0.0
            epochs_not_improved += 1

            for curr_x, curr_y in train_loader:
                curr_x, curr_y = curr_x.to(device), curr_y.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(curr_x.float())

                loss = self.loss_func(outputs, curr_y.float())
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Update learning rate scheduler
            scheduler.step()

            print(f"Epoch {epoch + 1}: Loss = {running_loss:.3f}")

            if running_loss < best_loss:
                best_loss = running_loss
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_not_improved = 0

            # Early stopping condition
            if epochs_not_improved >= MAX_EPOCHS_PATIENCE:
                print("Early stopping due to no improvement.")
                break

        # Restore the best model state
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
