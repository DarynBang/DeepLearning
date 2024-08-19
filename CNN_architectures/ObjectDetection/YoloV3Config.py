import torch
from torch import Tensor
import torch.nn as nn

"""
Information about architecture config:
Tuple is structured by (n_filters, kernel_size, stride)
Every conv is a same convolution.
Lists: "R" indicates a residual block followed by number of repeats.
"S" is for scale prediction block and computing yolo losses.
"U" is for upsampling the feature map and concatenating with the previous layer.
"""

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["R", 1],
    (128, 3, 2),
    ["R", 2],
    (256, 3, 2),
    ["R", 8],
    (512, 3, 2),
    ["R", 8],
    (1024, 3, 2),
    ["R", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S"
]



