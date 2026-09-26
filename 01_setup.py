# %%
# Run project scripts in numeric order in the same notebook/kernel namespace.
# Later cells reuse these imports and variables.
import torch
import torchvision
import torchvision.transforms.v2 as T
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
import optuna

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")
