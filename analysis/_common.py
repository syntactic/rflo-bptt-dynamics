"""Small conversion helpers shared across analysis channels."""

import numpy as np
import torch


def _to_numpy(x):
    """Convert a torch tensor (on any device) or array-like to a numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)
