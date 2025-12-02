"""
Shared model wrappers for inference interoperability.
"""

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is not available but is required for TorchModelWrapper. "
            "Install torch to use DNN models."
        )


if nn is not None:

    class TorchMLP(nn.Module):
        """Simple feed-forward network with ReLU activations."""

        def __init__(self, input_dim: int, hidden_sizes: Iterable[int]):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for size in hidden_sizes:
                layers.append(nn.Linear(prev, size))
                layers.append(nn.ReLU())
                prev = size
            layers.append(nn.Linear(prev, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[override]
            return self.network(x)
else:

    class TorchMLP:  # type: ignore[misc]
        pass


@dataclass
class TorchModelWrapper:
    """
    Wrapper that exposes predict_proba() compatible with scikit-learn.
    """

    input_dim: int
    hidden_sizes: List[int]
    state_dict: dict

    def __post_init__(self) -> None:
        _require_torch()
        self.model = TorchMLP(self.input_dim, self.hidden_sizes)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

    def predict_proba(self, features):
        _require_torch()
        arr = np.asarray(features, dtype=np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(arr)
            logits = self.model(tensor).reshape(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        probs = probs.astype(np.float64)
        stacked = np.vstack([1.0 - probs, probs]).T
        return stacked

