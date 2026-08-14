import gc
import os
import torch
import numpy as np

from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler
import src.utils as utils
import src.logging_utils as logging_utils


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
        use_cuda = getattr(device, "type", str(device)) == "cuda"
        use_amp = utils.env_flag("SPECTRA_AMP") and use_cuda
        use_channels_last = utils.env_flag("SPECTRA_CHANNELS_LAST") and use_cuda
        if use_channels_last:
            self.model.to(memory_format=torch.channels_last)

        correct = 0
        total = 0
        total_loss = 0.0
        n_batches = 0

        loss_func = self.loss_func if hasattr(self, 'loss_func') else torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                if use_channels_last and x_batch.dim() == 4:
                    x_batch = x_batch.contiguous(memory_format=torch.channels_last)
                if y_batch.dim() > 1 and y_batch.shape[1] > 1:  # one-hot targets
                    y_batch = torch.argmax(y_batch, dim=1)
                y_batch = y_batch.long()  # CrossEntropyLoss requires integer class indices
                with torch.cuda.amp.autocast(enabled=use_amp):
                    preds = self.model(x_batch)
                    batch_loss = loss_func(preds, y_batch)
                total_loss += batch_loss.item()
                pred_classes = torch.argmax(preds, dim=1)
                correct += int((pred_classes == y_batch).sum().item())
                total += int(y_batch.numel())
                n_batches += 1

        accuracy = (correct / total) if total else 0.0
        utils.print_flush(f"Accuracy: {accuracy:.3f}")
        utils.print_flush(f"Average Loss: {(total_loss / n_batches) if n_batches else 0.0:.3f}")
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
        use_cuda = getattr(device, "type", str(device)) == "cuda"
        use_amp = utils.env_flag("SPECTRA_AMP") and use_cuda
        use_channels_last = utils.env_flag("SPECTRA_CHANNELS_LAST") and use_cuda
        if use_channels_last:
            self.model.to(memory_format=torch.channels_last)
        if use_amp or use_channels_last:
            utils.print_flush(
                f"Fine-tune speed flags: AMP={int(use_amp)} channels_last={int(use_channels_last)}")

        num_epochs = conf.num_epochs
        if num_epochs <= 0:
            utils.print_flush("num_epochs <= 0; skipping post-compression fine-tuning.")
            return

        best_loss = np.inf
        best_state_buffer = None
        epochs_not_improved = 0
        # Early-stop patience for post-compression fine-tuning. NEON used 10; it was reduced
        # to 5 for short correctness runs and then starved recovery under a 1-epoch budget.
        # Override with SPECTRA_FINETUNE_PATIENCE. The epoch *budget* is conf.num_epochs
        # (40 by default, matching the SPECTRA argparse / NEON→40 comment).
        MAX_EPOCHS_PATIENCE = int(os.environ.get("SPECTRA_FINETUNE_PATIENCE", "10"))
        EPSILON = 1e-4

        # Recreate optimizer with current model parameters
        # Filter only trainable parameters to avoid non-grad tensors
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            utils.print_flush("No trainable parameters after freezing; skipping fine-tuning.")
            return
        # Frozen BatchNorm left in train() still updates running_mean/var from every batch,
        # which quietly destroys pretrained stats when train_compressed_layer_only is on.
        # Keep frozen norms in eval mode; trainable ones stay in train mode with the rest.
        frozen_norms = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                if not any(p.requires_grad for p in module.parameters(recurse=False)):
                    module.eval()
                    frozen_norms.append(module)
        if frozen_norms:
            utils.print_flush(
                f"BN-safe fine-tune: {len(frozen_norms)} frozen BatchNorm module(s) held in eval()")

        self.optimizer = torch.optim.Adam(trainable_params, lr=conf.learning_rate)
        # Clear any leftover optimizer state to ensure fresh start
        self.optimizer.state.clear()
        scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

        # Dynamic Learning Rate Scheduling  # TODO: New addition, assess with and without
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2)

        for epoch in range(num_epochs):  # 100 in NEON -> 40
            epoch_losses = []
            for curr_x, curr_y in train_loader:
                curr_x, curr_y = curr_x.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
                if use_channels_last and curr_x.dim() == 4:
                    curr_x = curr_x.contiguous(memory_format=torch.channels_last)

                # Skip batches with less than 2 samples to avoid issues in loss calculation
                if curr_x.size(0) < 2:
                    continue

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(curr_x)
                    if len(curr_y.shape) > 1 and curr_y.shape[1] > 1:
                        curr_y = torch.argmax(curr_y, dim=1)
                    loss = self.loss_func(outputs, curr_y.long())

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
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
                # Clone on-device; pickling to BytesIO every improving epoch was a CPU stall.
                best_state_buffer = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_not_improved = 0
            else:
                epochs_not_improved += 1

            # Full epoch traces at DEBUG; a short progress line every few epochs at INFO so a
            # 40-epoch fine-tune does not drown the run log in identical lines.
            if epoch == 0 or (epoch + 1) % 5 == 0 or epochs_not_improved == MAX_EPOCHS_PATIENCE:
                utils.print_flush(
                    f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.5f}, "
                    f"LR = {self.optimizer.param_groups[0]['lr']:.5f}")
            else:
                logging_utils.debug(
                    f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.5f}, "
                    f"LR = {self.optimizer.param_groups[0]['lr']:.5f}")

            if epochs_not_improved == MAX_EPOCHS_PATIENCE:
                utils.print_flush(
                    f"Early stopping at epoch {epoch + 1}/{num_epochs} "
                    f"(no improvement for {MAX_EPOCHS_PATIENCE} epochs; best_loss={best_loss:.5f})")
                break

        # `epoch` is defined after any non-empty training loop; empty-loader break leaves it unset
        try:
            epochs_ran = epoch + 1
        except NameError:
            epochs_ran = 0

        if best_state_buffer is not None and epochs_ran > 0 and epochs_not_improved < MAX_EPOCHS_PATIENCE:
            utils.print_flush(f"Fine-tune finished all {epochs_ran} epochs; best_loss={best_loss:.5f}")

        try:
            import src.run_recorder as _recorder
            _recorder.record(
                "finetune",
                epochs_budget=num_epochs,
                epochs_ran=epochs_ran,
                early_stopped=epochs_not_improved >= MAX_EPOCHS_PATIENCE,
                best_loss=None if best_loss == np.inf else round(float(best_loss), 6),
                patience=MAX_EPOCHS_PATIENCE,
            )
        except Exception:
            pass

        # If training fails to converge - reinitializing weights and retraining (at most once)
        if best_loss == np.inf:
            if allow_reinit_retry:
                utils.print_flush("Model failed to converge. Reinitializing weights and retrying once.")
                self.reinitialize_weights()
                return self.train_model(train_loader, allow_reinit_retry=False)
            utils.print_flush("Model failed to converge after a reinitialisation retry; keeping current weights.")
        elif best_state_buffer is not None:
            self.model.load_state_dict(best_state_buffer)

        # Free up cache and memory after training. Identity-heavy evals spend a lot of
        # wall time in empty_cache; SPECTRA_SKIP_FT_GC=1 is the experimental speed arm.
        del self.optimizer
        if not utils.env_flag("SPECTRA_SKIP_FT_GC"):
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
