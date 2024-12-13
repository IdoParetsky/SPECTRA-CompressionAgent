import copy

import torch
from sklearn.metrics import accuracy_score
import numpy as np

from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler


class Dataset(torch.utils.data.Dataset):
    def int_to_onehot(self, idx):
        one_hot = torch.zeros(self.range_y).float()
        one_hot[int(idx) - int(self.min_y)] = 1.0  # Adjusting class labels to be zero-based
        return one_hot

    def __init__(self, x, y):
        self.min_y = min(y)
        self.max_y = max(y)
        self.range_y = int(self.max_y - self.min_y + 1)

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.int_to_onehot(self.y[idx])


class ClassificationHandler(BasicHandler):

    def evaluate_model(self, loader) -> float:
        """
        Evaluates the model's performance.

        Args:
            loader (DataLoader): The DataLoader for the validation or test set.

        Returns:
            float: The accuracy score of the model.
        """
        self.model.eval()
        device = StaticConf.get_instance().conf_values.device
        self.model.to(device)

        all_preds = []
        all_labels = []

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = self.model(x_batch).detach()
            preds_classes = torch.argmax(preds, dim=1)

            all_preds.extend(preds_classes.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def train_model(self, train_loader):
        """
         Trains the model using the provided training data loader.

         Args:
             train_loader (DataLoader): The DataLoader for training
         """
        device = StaticConf.get_instance().conf_values.device
        self.model.float().to(device)
        self.model.train()

        best_loss = np.inf
        best_state_dict = None
        epochs_not_improved = 0
        MAX_EPOCHS_PATIENCE = 10  # TODO: Consider updating / dynamically changing

        # Ensure optimizer is configured with current model parameters
        self.optimizer.param_groups[0]['params'] = list(self.model.parameters())

        for epoch in range(StaticConf.get_instance().conf_values.num_epoch):
            running_loss = 0.0
            epochs_not_improved += 1

            for i, (curr_x, curr_y) in enumerate(train_loader):
                curr_x, curr_y = curr_x.to(device), curr_y.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(curr_x.float())

                # Ensure curr_y is processed correctly for classification
                if len(curr_y.shape) > 1 and curr_y.shape[1] > 1:  # One-hot encoded labels
                    curr_y = torch.argmax(curr_y, dim=1)

                loss = self.loss_func(outputs, curr_y)

                if loss < best_loss:
                    epochs_not_improved = 0
                    best_loss = loss
                    best_state_dict = copy.deepcopy(self.model.state_dict())

                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}: Loss = {running_loss:.4f}")

            if epochs_not_improved == MAX_EPOCHS_PATIENCE:
                print("Early stopping due to no improvement.")
                break

        # Restore the best model state
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
