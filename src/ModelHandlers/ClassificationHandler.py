import gc
import math
import os
import torch
import torch.nn.functional as F
import numpy as np

from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.BasicHandler import BasicHandler
import src.utils as utils
import src.logging_utils as logging_utils


def _ft_float(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def mixup_batch(x, y, alpha: float):
    """MixUp on a class-index batch. ``alpha <= 0`` is a no-op. Returns x, y_a, y_b, lam."""
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, None, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[index]
    return mixed, y, y[index], lam


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

    def __init__(self, model, loss_function):
        super().__init__(model, loss_function)
        self.kd_teacher = None

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

    def train_model(self, train_loader, allow_reinit_retry=True, max_epochs=None, patience=None,
                    val_loader=None, lr_mult=1.0, tag=""):
        """
         Fine-tunes the model after a compression step, keeping the best-loss weights.

         Args:
             train_loader (DataLoader): The DataLoader for training
             allow_reinit_retry (bool): Whether a non-converging run may reinitialise the
                 weights and train once more. The retry is single-shot on purpose: it used
                 to call train_model unconditionally, so any configuration that produced no
                 epoch loss at all (e.g. num_epochs == 0, or an empty loader) recursed until
                 the interpreter hit its recursion limit.
             max_epochs (int, optional): Epoch budget for this call instead of
                 ``conf.num_epochs`` (policy-training recovery, SPECTRA_TRAIN_FT_EPOCHS).
             patience (int, optional): Early-stop patience for this call instead of
                 ``SPECTRA_FINETUNE_PATIENCE``.
             val_loader (DataLoader, optional): P8 (NEON layer replacement). When given, the
                 epoch selection and the patience run on **val accuracy** ("train the new
                 layer until convergence") instead of the train loss; the best-val state is
                 restored. Default None keeps the live train-loss selection byte-identical.
             lr_mult (float): Multiplier on the fine-tune learning rate (C-G+ polish uses 0.1).
             tag (str): Log prefix for multi-phase recipes (``"C-G group"`` / ``"C-G+ polish"``).
         """
        conf = StaticConf.get_instance().conf_values
        device = conf.device
        select_on_val = val_loader is not None
        log_tag = f"[{tag}] " if tag else ""
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

        num_epochs = int(max_epochs) if max_epochs is not None else conf.num_epochs
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
        MAX_EPOCHS_PATIENCE = (int(patience) if patience is not None
                               else int(os.environ.get("SPECTRA_FINETUNE_PATIENCE", "10")))
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

        optim_name = os.environ.get("SPECTRA_FT_OPTIM", "adam").strip().lower() or "adam"
        mixup_alpha = _ft_float("SPECTRA_FT_MIXUP", "0")
        label_smooth = _ft_float("SPECTRA_FT_LABEL_SMOOTH", "0")
        use_cosine = utils.env_flag("SPECTRA_FT_COSINE")
        kd_t = _ft_float("SPECTRA_FT_KD_T", "4")
        kd_alpha = _ft_float("SPECTRA_FT_KD_ALPHA", "0.7")
        use_kd = self.kd_teacher is not None and utils.env_flag("SPECTRA_FT_KD")
        if label_smooth > 0:
            self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=label_smooth)
        if use_kd:
            self.kd_teacher.to(device).eval()
            for param in self.kd_teacher.parameters():
                param.requires_grad = False
            if use_channels_last:
                self.kd_teacher.to(memory_format=torch.channels_last)

        lr_mult = float(lr_mult) if lr_mult else 1.0
        weight_decay = _ft_float("SPECTRA_FT_WD", "5e-4")
        # V7 one-recipe schedule (default off): SPECTRA_FT_SCHEDULE=warmcos = one epoch of
        # linear warmup to the peak, then cosine down to SPECTRA_FT_LR_MIN over the remaining
        # epochs, stepped per batch. Adam's early steps have exploding variance (Liu et al.,
        # ICLR 2020, RAdam); warmup is the variance reducer that a constant 1e-3 lacks and a
        # constant 1e-4 over-corrects. Pair with AdamW (decoupled decay, Loshchilov & Hutter,
        # ICLR 2019) so the adaptive arm has the decay the SGD arm already had.
        schedule = os.environ.get("SPECTRA_FT_SCHEDULE", "").strip().lower()
        use_warmcos = schedule == "warmcos"
        if optim_name == "sgd":
            sgd_lr = _ft_float("SPECTRA_FT_SGD_LR", "0.01") * lr_mult
            momentum = _ft_float("SPECTRA_FT_MOMENTUM", "0.9")
            self.optimizer = torch.optim.SGD(
                trainable_params, lr=sgd_lr, momentum=momentum, weight_decay=weight_decay)
            shown_lr = sgd_lr
        elif optim_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params, lr=conf.learning_rate * lr_mult, weight_decay=weight_decay)
            shown_lr = conf.learning_rate * lr_mult
        elif optim_name == "radam":
            self.optimizer = torch.optim.RAdam(
                trainable_params, lr=conf.learning_rate * lr_mult, weight_decay=weight_decay)
            shown_lr = conf.learning_rate * lr_mult
        else:
            self.optimizer = torch.optim.Adam(trainable_params, lr=conf.learning_rate * lr_mult)
            shown_lr = conf.learning_rate * lr_mult
        self.optimizer.state.clear()
        scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None
        steps_per_epoch = max(1, len(train_loader)) if hasattr(train_loader, "__len__") else 1
        if use_warmcos:
            lr_min = _ft_float("SPECTRA_FT_LR_MIN", "1e-5")
            warm_steps = max(1, int(round(_ft_float("SPECTRA_FT_WARMUP_EPOCHS", "1") * steps_per_epoch)))
            total_steps = max(warm_steps + 1, num_epochs * steps_per_epoch)
            floor = lr_min / max(shown_lr, 1e-12)

            def _warmcos(step):
                if step < warm_steps:
                    return max(floor, (step + 1) / warm_steps)
                progress = min(1.0, (step - warm_steps) / max(1, total_steps - warm_steps))
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _warmcos)
        elif use_cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(num_epochs, 1))
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2)
        best_val = -np.inf
        n_trainable = sum(p.numel() for p in trainable_params)
        utils.print_flush(
            f"{log_tag}Fine-tune recipe: optim={optim_name} lr={shown_lr:g} cosine={int(use_cosine)} "
            f"schedule={'warmcos' if use_warmcos else 'plateau' if not use_cosine else 'cosine'} "
            f"wd={weight_decay if optim_name in ('sgd', 'adamw', 'radam') else 0:g} "
            f"mixup={mixup_alpha:g} smooth={label_smooth:g} kd={int(use_kd)} "
            f"patience={MAX_EPOCHS_PATIENCE} epochs={num_epochs} "
            f"select={'val' if select_on_val else 'train_loss'} trainable={n_trainable}")

        for epoch in range(num_epochs):  # 100 in NEON -> 40
            epoch_losses = []
            for curr_x, curr_y in train_loader:
                curr_x, curr_y = curr_x.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
                if use_channels_last and curr_x.dim() == 4:
                    curr_x = curr_x.contiguous(memory_format=torch.channels_last)

                # Skip batches with less than 2 samples to avoid issues in loss calculation
                if curr_x.size(0) < 2:
                    continue
                if len(curr_y.shape) > 1 and curr_y.shape[1] > 1:
                    curr_y = torch.argmax(curr_y, dim=1)

                curr_x, y_a, y_b, lam = mixup_batch(curr_x, curr_y, mixup_alpha)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(curr_x)
                    if y_b is None:
                        ce = self.loss_func(outputs, y_a.long())
                    else:
                        ce = lam * self.loss_func(outputs, y_a.long()) + (
                            1.0 - lam) * self.loss_func(outputs, y_b.long())
                    if use_kd:
                        with torch.no_grad():
                            teacher_logits = self.kd_teacher(curr_x)
                        log_s = F.log_softmax(outputs / kd_t, dim=1)
                        p_t = F.softmax(teacher_logits / kd_t, dim=1)
                        kd = F.kl_div(log_s, p_t, reduction="batchmean") * (kd_t * kd_t)
                        loss = (1.0 - kd_alpha) * ce + kd_alpha * kd
                    else:
                        loss = ce

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
                if use_warmcos:
                    scheduler.step()

                epoch_losses.append(loss.detach())

            if not epoch_losses:
                utils.print_flush("Training loader yielded no usable batches; aborting fine-tuning.")
                break

            avg_loss = torch.stack(epoch_losses).mean().item()
            if use_warmcos:
                pass  # stepped per batch above
            elif use_cosine:
                scheduler.step()
            else:
                scheduler.step(avg_loss)

            if select_on_val:
                # P8: "train the new layer until convergence" — measured where the reward is
                # measured. evaluate_model flips eval(); restore train mode for the frozen-BN
                # bookkeeping (trainable modules back to train, frozen norms stay in eval).
                val_acc = float(self._quiet_val_accuracy(val_loader, device))
                self.model.train()
                for module in frozen_norms:
                    module.eval()
                improved = val_acc > best_val + EPSILON
                if improved:
                    best_val = val_acc
                if avg_loss < best_loss:
                    best_loss = avg_loss
            else:
                val_acc = None
                improved = avg_loss < best_loss - EPSILON
                if improved:
                    best_loss = avg_loss
            if improved:
                # Clone on-device; pickling to BytesIO every improving epoch was a CPU stall.
                best_state_buffer = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_not_improved = 0
            else:
                epochs_not_improved += 1

            # Full epoch traces at DEBUG; a short progress line every few epochs at INFO so a
            # 40-epoch fine-tune does not drown the run log in identical lines.
            val_note = f", val_acc = {val_acc:.4f}" if val_acc is not None else ""
            if epoch == 0 or (epoch + 1) % 5 == 0 or epochs_not_improved == MAX_EPOCHS_PATIENCE:
                utils.print_flush(
                    f"{log_tag}Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.5f}{val_note}, "
                    f"LR = {self.optimizer.param_groups[0]['lr']:.5f}")
            else:
                logging_utils.debug(
                    f"{log_tag}Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.5f}{val_note}, "
                    f"LR = {self.optimizer.param_groups[0]['lr']:.5f}")

            if epochs_not_improved == MAX_EPOCHS_PATIENCE:
                best_note = (f"best_val={best_val:.4f}" if select_on_val
                             else f"best_loss={best_loss:.5f}")
                utils.print_flush(
                    f"{log_tag}Early stopping at epoch {epoch + 1}/{num_epochs} "
                    f"(no improvement for {MAX_EPOCHS_PATIENCE} epochs; {best_note})")
                break

        # `epoch` is defined after any non-empty training loop; empty-loader break leaves it unset
        try:
            epochs_ran = epoch + 1
        except NameError:
            epochs_ran = 0

        if best_state_buffer is not None and epochs_ran > 0 and epochs_not_improved < MAX_EPOCHS_PATIENCE:
            utils.print_flush(f"{log_tag}Fine-tune finished all {epochs_ran} epochs; best_loss={best_loss:.5f}")

        try:
            import src.run_recorder as _recorder
            _recorder.record(
                "finetune",
                epochs_budget=num_epochs,
                epochs_ran=epochs_ran,
                early_stopped=epochs_not_improved >= MAX_EPOCHS_PATIENCE,
                best_loss=None if best_loss == np.inf else round(float(best_loss), 6),
                patience=MAX_EPOCHS_PATIENCE,
                select="val" if select_on_val else "train_loss",
                best_val=None if best_val == -np.inf else round(float(best_val), 5),
                phase=tag or None,
                trainable_params=n_trainable,
            )
        except Exception:
            pass

        # Empty loader used to Xavier-reinit the whole CNN and destroy the pruned net.
        if best_loss == np.inf:
            utils.print_flush(
                "Fine-tune produced no loss (empty loader); keeping pruned weights.")
        elif best_state_buffer is not None:
            self.model.load_state_dict(best_state_buffer)

        # Free up cache and memory after training. Identity-heavy evals spend a lot of
        # wall time in empty_cache; SPECTRA_SKIP_FT_GC=1 is the experimental speed arm.
        del self.optimizer
        if not utils.env_flag("SPECTRA_SKIP_FT_GC"):
            torch.cuda.empty_cache()
            gc.collect()

    def _quiet_val_accuracy(self, loader, device) -> float:
        """Val accuracy without the per-call log lines (per-epoch selection inside train_model)."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                if y_batch.dim() > 1 and y_batch.shape[1] > 1:
                    y_batch = torch.argmax(y_batch, dim=1)
                preds = torch.argmax(self.model(x_batch), dim=1)
                correct += int((preds == y_batch.long()).sum().item())
                total += int(y_batch.numel())
        return (correct / total) if total else 0.0

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
