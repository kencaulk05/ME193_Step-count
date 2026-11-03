"""Other architectures, ONLY for the comparison in arch_compare.py. The shipped counter is steps_model.StepNet.

Every variant has the same job and the same interface: signal (batch, 8, time) -> step density (batch, time), never negative, whose sum is the count.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from steps_data import N_FEATURES
from steps_model import INIT_BIAS, StepNet


class StepGRU(nn.Module):
    """A bidirectional GRU instead of convolutions. A recurrent network is slow on a 100 Hz signal, so the signal is first averaged down
    to 100 / pool readings per second, and the density is spread back over the readings, so the sum (the count) is unchanged."""

    def __init__(self, hidden=16, pool=8, n_in=N_FEATURES, init_bias=INIT_BIAS):
        super().__init__()
        self.pool = pool
        self.gru = nn.GRU(n_in, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)
        nn.init.constant_(self.head.bias, init_bias + math.log(pool))   # the same starting density PER READING as the convolutional networks

    def forward(self, x):
        T = x.shape[2]
        p = F.avg_pool1d(x, self.pool, ceil_mode=True)               # (batch, 8, ceil(T / pool))
        h, _ = self.gru(p.transpose(1, 2))
        d = F.softplus(self.head(h)).squeeze(-1)                     # steps per pooled moment
        return d.repeat_interleave(self.pool, dim=1)[:, :T] / self.pool


def receptive_field(kernel, dilations):
    return 1 + (kernel - 1) * sum(dilations)


# name -> (what it is, builder, batch size). The first is the shipped network, kept as the reference.
VARIANTS = {
    "reference": ("shipped: 16 filters, kernel 5, dilations 1-16 (5 layers)", lambda: StepNet(), 4),
    "narrow": ("8 filters instead of 16", lambda: StepNet(width=8), 4),
    "wide": ("32 filters instead of 16", lambda: StepNet(width=32), 4),
    "shallow": ("3 layers, dilations 1, 4, 16 (reach 0.85 s)", lambda: StepNet(dilations=(1, 4, 16)), 4),
    "deep": ("6 layers, dilations 1-32 (reach 2.5 s)", lambda: StepNet(dilations=(1, 2, 4, 8, 16, 32)), 4),
    "kernel3": ("kernel 3, dilations 1-32 (same reach, 1.27 s)", lambda: StepNet(kernel=3, dilations=(1, 2, 4, 8, 16, 32)), 4),
    "kernel9": ("kernel 9, dilations 1-8 (reach 1.2 s)", lambda: StepNet(kernel=9, dilations=(1, 2, 4, 8)), 4),
    "gru": ("bidirectional GRU, 16 units, on the signal averaged to 12.5 Hz", lambda: StepGRU(), 1),
}
