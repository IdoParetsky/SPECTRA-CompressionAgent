import copy

import torch
from sklearn.metrics import accuracy_score
import numpy as np

from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler
import src.utils as utils


# TODO: Consider data normalization and augmentation via torchvision.transforms
class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.min_y = min(y)
        self.max_y = max(y)
        self.range_y = int(self.max_y - self.min_y + 1)

        self.x = torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x
        self.y = torch.tensor(y, dtype=torch.float32) if isinstance(y, np.ndarray) else y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
        total_loss = 0.0

        loss_func = self.loss_func if hasattr(self, 'loss_func') else torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = self.model(x_batch)
                total_loss += loss_func(preds, y_batch).item()
                preds_classes = torch.argmax(preds, dim=1)

                all_preds.extend(preds_classes.detach().tolist())
                all_labels.extend(y_batch.detach().tolist())

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        utils.print_flush(f"Accuracy: {accuracy:.3f}")
        utils.print_flush(f"Average Loss: {total_loss / len(loader):.3f}")
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

        # Dynamic Learning Rate Scheduling  # TODO: New addition, assess with and without
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        for epoch in range(StaticConf.get_instance().conf_values.num_epochs):
            running_loss = 0.0
            epochs_not_improved += 1

            for i, (curr_x, curr_y) in enumerate(train_loader):
                curr_x, curr_y = curr_x.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)

                # Skip batches with less than 2 samples to avoid issues in loss calculation
                if curr_x.size(0) < 2:
                    continue

                self.optimizer.zero_grad()
                curr_x.requires_grad_(True)
                outputs = self.model(curr_x)

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

                running_loss += loss.item()

            # Step the scheduler after every epoch based on validation loss
            scheduler.step(running_loss / len(train_loader))

            utils.print_flush(f"Epoch {epoch + 1}: Loss = {running_loss / len(train_loader):.3f}, "
                  f"Learning Rate = {self.optimizer.param_groups[0]['lr']:.3f}")

            if epochs_not_improved == MAX_EPOCHS_PATIENCE:
                utils.print_flush("Early stopping due to no improvement.")
                break

        # If training fails to converge - reinitializing weights and retraining
        if best_loss == np.inf:
            utils.print_flush("Model failed to converge. Reinitializing weights.")
            self.reinitialize_weights()
            return self.train_model(train_loader)  # Retry training

        # Restore the best model state
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

    def reinitialize_weights(self):
        """
        Reinitializes the model's weights using Xavier or He initialization,
        depending on the activation function.
        """
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.model.apply(init_weights)
