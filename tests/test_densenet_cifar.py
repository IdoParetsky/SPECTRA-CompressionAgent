"""DenseNet-BC-40 (CIFAR stem) is in the thesis architecture pool; concat groups must prune."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402


def test_densenet40_cifar_prunes_and_runs():
    from spectra_models_instantiation.densenet_cifar import densenet40

    model = densenet40(num_classes=10, large_input=False).eval()
    sample = torch.randn(1, 3, 32, 32)
    params_before = sum(p.numel() for p in model.parameters())

    num_rows = len(ModelWithRows(model).row_to_main_layer)
    for row_idx in range(num_rows):
        model_with_rows = ModelWithRows(model)
        prune_current_model(model_with_rows, 0.8, row_idx)
        model = model_with_rows.model

    out = model.eval()(sample)
    assert out.shape == (1, 10)
    assert sum(p.numel() for p in model.parameters()) < params_before
