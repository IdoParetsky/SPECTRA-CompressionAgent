import io
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import torch.distributed as dist

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
                if y_batch.dim() > 1 and y_batch.shape[1] > 1:  # one-hot targets
                    y_batch = torch.argmax(y_batch, dim=1)
                y_batch = y_batch.long()  # CrossEntropyLoss requires integer class indices
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

    def train_model(self, train_loader, allow_reinit_retry=True):
        """
         Fine-tunes the model after a compression step, keeping the best-loss weights.

         Args:
             train_loader (DataLoader): The DataLoader for training
             allow_reinit_retry (bool): Whether a non-converging run may reinitialise the
                 weights and train once more. The retry is single-shot on purpose: it used
                 to call train_model unconditionally, so any configuration that produced no
                 epoch loss at all (e.g. num_epochs == 0, or an empty loader) recursed until
                 the interpreter hit its recursion limit.
         """
        conf = StaticConf.get_instance().conf_values
        device = conf.device
        self.model.float().to(device)
        self.model.train()

        num_epochs = conf.num_epochs
        if num_epochs <= 0:
            utils.print_flush("num_epochs <= 0; skipping post-compression fine-tuning.")
            return

        best_loss = np.inf
        best_state_buffer = None
        epochs_not_improved = 0
        MAX_EPOCHS_PATIENCE = 5  # Changed from 10 in NEON. TODO: Consider updating / dynamically changing
        EPSILON = 1e-4

        # Recreate optimizer with current model parameters
        # Filter only trainable parameters to avoid non-grad tensors
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            utils.print_flush("No trainable parameters after freezing; skipping fine-tuning.")
            return
        self.optimizer = torch.optim.Adam(trainable_params, lr=conf.learning_rate)
        # Clear any leftover optimizer state to ensure fresh start
        self.optimizer.state.clear()

        # Dynamic Learning Rate Scheduling  # TODO: New addition, assess with and without
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2)

        for epoch in range(num_epochs):  # 100 in NEON -> 40
            epoch_losses = []
            for curr_x, curr_y in train_loader:
                curr_x, curr_y = curr_x.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)

                # Skip batches with less than 2 samples to avoid issues in loss calculation
                if curr_x.size(0) < 2:
                    continue

                self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(curr_x)
                if len(curr_y.shape) > 1 and curr_y.shape[1] > 1:
                    curr_y = torch.argmax(curr_y, dim=1)
                loss = self.loss_func(outputs, curr_y.long())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                self.optimizer.step()

                epoch_losses.append(loss.detach())

            if not epoch_losses:
                utils.print_flush("Training loader yielded no usable batches; aborting fine-tuning.")
                break

            avg_loss = torch.stack(epoch_losses).mean().item()
            scheduler.step(avg_loss)

            if avg_loss < best_loss - EPSILON:
                best_loss = avg_loss
                # Every rank keeps its own copy so that non-zero ranks can restore too
                best_state_buffer = io.BytesIO()
                torch.save(self.model.state_dict(), best_state_buffer)
                epochs_not_improved = 0
            else:
                epochs_not_improved += 1

            utils.print_flush(f"Epoch {epoch + 1}: Loss = {avg_loss:.5f}, "
                              f"Learning Rate = {self.optimizer.param_groups[0]['lr']:.5f}")

            if epochs_not_improved == MAX_EPOCHS_PATIENCE:
                utils.print_flush("Early stopping due to no improvement.")
                break

        # If training fails to converge - reinitializing weights and retraining (at most once)
        if best_loss == np.inf:
            if allow_reinit_retry:
                utils.print_flush("Model failed to converge. Reinitializing weights and retrying once.")
                self.reinitialize_weights()
                return self.train_model(train_loader, allow_reinit_retry=False)
            utils.print_flush("Model failed to converge after a reinitialisation retry; keeping current weights.")
        elif best_state_buffer is not None:
            # Restore the best model state
            best_state_buffer.seek(0)
            self.model.load_state_dict(torch.load(best_state_buffer, weights_only=True, map_location=device))

        # Free up cache and memory after training
        del self.optimizer
        torch.cuda.empty_cache()
        gc.collect()

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
