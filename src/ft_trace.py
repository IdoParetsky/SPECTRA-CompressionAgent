"""
TRACE (``SPECTRA_EVAL_FT_TRACE=<K>``, default 0 = off): a read along the final fine-tune.

:class:`FtTrace` is the ``step_hook`` of ``ClassificationHandler.train_model``. At step 0, after every K-th
optimizer step and at the last step it reads the val loader's accuracy and mean cross-entropy loss, and reports
the train loss and the clipped gradient's global norm averaged since the previous read, the learning rate, and
each convolution's relative weight movement ``||W - W0|| / ||W0||`` (mean, max, first and last layer; ``W0`` is
the weight at step 0, the cut's own).

Training is unchanged by a read. It saves and restores the torch (CPU and CUDA), ``random`` and numpy random
states (a ``DataLoader`` iterator draws a base seed from the global torch generator even without shuffling),
and each module's ``training`` flag (set directly, so a BatchNorm that ``train_model`` holds in eval stays
there). The accumulators take the loss and the gradient norm as device tensors, with no ``.item()`` between
reads. The read mirrors ``ClassificationHandler.evaluate_model`` (device moves, AMP autocast, channels_last,
argmax of one-hot targets) and prints nothing per batch. Val only, never TEST.

    [eval] FT_TRACE <label> <net> step=<s> | val <acc> loss <l> | train loss <l> | gnorm <g> | lr <lr> |
        wmove mean <m> max <M> | <s> s

NAP (``SPECTRA_EVAL_FT_NAP=1``, :mod:`src.ft_nap`) rides this hook through the keyword ``nap``: ``on_start`` calls it
after the step-0 read, ``on_step`` before the read logic and ``on_end`` after the last read. With ``nap=None`` (the
default) nothing changes.
"""

import random
import time

import numpy as np
import torch

import src.utils as utils


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


class FtTrace:
    def __init__(self, every, val_loader, device, label, name, network, recorder=None, printer=None, nap=None):
        self.every = max(1, int(every))
        self.val_loader = val_loader
        self.device = device
        self.label = label
        self.name = name
        self.network = network
        self.recorder = recorder
        self.printer = printer
        self.nap = nap  # NAP (SPECTRA_EVAL_FT_NAP, src/ft_nap.py) rides this hook; None (the default) changes nothing
        self.records = []
        self.model = None
        self.convs = []
        self.w0 = []
        self.final_done = False
        self._last_read_step = None
        self._lr = None
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._loss_sum = None
        self._gnorm_sum = None
        self._count = 0

    def on_start(self, model):
        self.model = model
        self.convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        self.w0 = [m.weight.detach().clone() for m in self.convs]
        self._reset_accumulators()
        self._lr = None
        self._read(0)
        if self.nap is not None:
            self.nap.on_start(model)

    def on_step(self, step, loss, grad_norm, lr):
        if self.nap is not None:
            self.nap.on_step(step, loss, grad_norm, lr)
        loss = loss.detach().float()
        grad_norm = grad_norm.detach().float()
        self._loss_sum = loss if self._loss_sum is None else self._loss_sum + loss
        self._gnorm_sum = grad_norm if self._gnorm_sum is None else self._gnorm_sum + grad_norm
        self._count += 1
        self._lr = lr
        if step % self.every == 0:
            self._read(step)

    def on_end(self, step):
        if self._last_read_step != step:
            self._read(step)
        self.final_done = True
        if self.nap is not None:
            self.nap.on_end(step)

    # --- the read -------------------------------------------------------------------------------------

    def _val_read(self):
        """Val accuracy and mean cross-entropy, with ``evaluate_model``'s numerics; no per-batch output."""
        model = self.model
        device = self.device
        use_cuda = getattr(device, "type", str(device)) == "cuda"
        use_amp = utils.env_flag("SPECTRA_AMP") and use_cuda
        use_channels_last = utils.env_flag("SPECTRA_CHANNELS_LAST") and use_cuda
        correct = None
        loss_sum = None
        total = 0
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                if use_channels_last and x_batch.dim() == 4:
                    x_batch = x_batch.contiguous(memory_format=torch.channels_last)
                if y_batch.dim() > 1 and y_batch.shape[1] > 1:  # one-hot targets
                    y_batch = torch.argmax(y_batch, dim=1)
                y_batch = y_batch.long()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(x_batch)
                batch_correct = (torch.argmax(outputs, dim=1) == y_batch).sum()
                batch_loss = torch.nn.functional.cross_entropy(outputs.float(), y_batch, reduction="sum")
                correct = batch_correct if correct is None else correct + batch_correct
                loss_sum = batch_loss if loss_sum is None else loss_sum + batch_loss
                total += int(y_batch.numel())
        if not total:
            return 0.0, 0.0
        return int(correct.item()) / total, float(loss_sum.item()) / total

    @staticmethod
    def _save_rng():
        return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                random.getstate(), np.random.get_state())

    @staticmethod
    def _restore_rng(states):
        torch_state, cuda_state, py_state, np_state = states
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        random.setstate(py_state)
        np.random.set_state(np_state)

    def _read(self, step):
        t0 = time.perf_counter()
        model = self.model
        rng_states = self._save_rng()
        flags = [(m, m.training) for m in model.modules()]
        try:
            val_acc, val_loss = self._val_read()
        finally:
            for module, training in flags:
                module.training = training  # the attribute, not train(): train() recurses
            self._restore_rng(rng_states)

        train_loss = grad_norm = None
        if self._count:
            train_loss = float(self._loss_sum.item()) / self._count
            grad_norm = float(self._gnorm_sum.item()) / self._count
        wmove = self._weight_movement()
        seconds = time.perf_counter() - t0
        record = dict(
            step=int(step), val_acc=float(val_acc), val_loss=float(val_loss), train_loss=train_loss,
            grad_norm=grad_norm, lr=None if self._lr is None else float(self._lr),
            wmove_mean=wmove["mean"], wmove_max=wmove["max"], wmove_first=wmove["first"],
            wmove_last=wmove["last"], n_conv=len(self.convs), seconds=round(seconds, 3))
        self._reset_accumulators()
        self._last_read_step = step
        self.records.append(record)
        if self.recorder is not None:
            self.recorder.record("eval_ft_trace", network=self.network, label=self.label, **record)
        if self.printer is not None:
            self.printer(
                f"[eval] FT_TRACE {self.label} {self.name} step={step} | val {record['val_acc']:.4f} "
                f"loss {record['val_loss']:.4f} | train loss {_fmt(train_loss, '.4f')} | "
                f"gnorm {_fmt(grad_norm, '.4f')} | lr {_fmt(record['lr'], 'g')} | "
                f"wmove mean {_fmt(wmove['mean'], '.4f')} max {_fmt(wmove['max'], '.4f')} | {seconds:.1f} s")

    def _weight_movement(self):
        if not self.convs:
            return {"mean": None, "max": None, "first": None, "last": None}
        with torch.no_grad():
            moved = torch.stack([(m.weight.detach() - w0).float().norm() for m, w0 in zip(self.convs, self.w0)])
            base = torch.stack([w0.float().norm() for w0 in self.w0])
            moved, base = moved.tolist(), base.tolist()
        rel = [(d / b) if b > 0.0 else 0.0 for d, b in zip(moved, base)]
        return {"mean": sum(rel) / len(rel), "max": max(rel), "first": rel[0], "last": rel[-1]}
