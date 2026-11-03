"""The step counter, in PyTorch.

WHAT THE NETWORK DOES
    It reads the ankle IMU signal (8 numbers every 10 ms) and, at every moment, outputs one number: how many steps are happening right now
    (a "step density", steps per reading, never negative). Add the density up over a whole trial and you get the step count, both feet.

WHAT IT LEARNS FROM
    Only the number you counted for each trial. There is no label saying WHERE each step is. The loss is how far the sum is from your count,
    so the network has to discover for itself what a step looks like in the signal, and that standing, tapping a foot and air-biking are not steps.

    forward pass:   signal -> 5 convolution layers (each with a ReLU) -> softplus -> density per moment
    loss:           | ln(1 + sum of the density) - ln(1 + the steps you counted) |   (about the RELATIVE error on a long walk, about the plain
                    error on a short one, and gentle on trials with 0 steps)
    backward pass:  backpropagation finds how each of the ~6000 weights changed that error; Adam nudges every weight a little the other way
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from steps_data import FS, N_FEATURES, features
from steps_numpy import NumpyStepNet

WIDTH = 16
KERNEL = 5
DILATIONS = (1, 2, 4, 8, 16)                 # receptive field 1 + 4 x (1+2+4+8+16) = 125 readings = 1.25 seconds: long enough to see a whole stride
INIT_BIAS = -7.0                             # the density starts at softplus(-7) = 0.0009 per reading. Starting at -4 (0.018, so 36 false steps in a
                                             # 20-second trial with no steps) made training collapse to "count nothing" in 2 of 5 runs on the real trials
COLLAPSE_RATIO = 0.3                         # a network that counts less than this fraction of the true steps on ITS OWN training walks is stuck counting nothing
                                             # (stuck: about 0%, healthy: 93-100%). The training loss is not used: a healthy run can still be improving slowly.


class StepNet(nn.Module):
    def __init__(self, width=WIDTH, kernel=KERNEL, dilations=DILATIONS, n_in=N_FEATURES, init_bias=INIT_BIAS):
        super().__init__()
        layers, c = [], n_in
        for d in dilations:
            layers += [nn.Conv1d(c, width, kernel, padding=d * (kernel - 1) // 2, dilation=d), nn.ReLU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(width, 1, 1)
        nn.init.constant_(self.head.bias, init_bias)             # start with a tiny density: about 0.001 steps per reading, 0.1 a second
        self.dilations, self.kernel = tuple(dilations), kernel

    def forward(self, x):                                        # x: (batch, 8, time) -> density (batch, time)
        return F.softplus(self.head(self.body(x))).squeeze(1)

    def to_numpy(self, meta=None):
        convs = [m for m in self.body if isinstance(m, nn.Conv1d)]
        layers = [{"w": c.weight.detach().cpu().numpy(), "b": c.bias.detach().cpu().numpy(), "dilation": d} for c, d in zip(convs, self.dilations)]
        return NumpyStepNet(layers, self.head.weight.detach().cpu().numpy(), self.head.bias.detach().cpu().numpy(), meta)


# ---- augmentation: changes that do NOT change how many steps there were ---------------------------------------------------
def _rotation(rng, max_deg):
    ax, ay, az = np.deg2rad(rng.uniform(-max_deg, max_deg, 3))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def augment(x, rng):
    """A new version of the same trial with the same number of steps:
    the sensor sitting a little differently on the ankle (a small rotation), a different strength, the same steps a bit faster or slower,
    and a little more noise."""
    x = np.asarray(x, dtype=np.float64)
    R = _rotation(rng, 12.0)
    x = np.concatenate([x[:, :3] @ R.T, x[:, 3:] @ R.T], axis=1)
    x[:, :3] *= rng.uniform(0.85, 1.15)
    x[:, 3:] *= rng.uniform(0.85, 1.15)
    T = len(x)
    new_T = max(int(T * rng.uniform(0.8, 1.25)), 50)
    idx = np.linspace(0, T - 1, new_T)
    x = np.stack([np.interp(idx, np.arange(T), x[:, c]) for c in range(6)], axis=1)
    x[:, :3] += rng.normal(0, 0.01, (new_T, 3))
    x[:, 3:] += rng.normal(0, 1.0, (new_T, 3))
    return x.astype(np.float32)


def _batch(trials, rng, aug):
    xs = [features(augment(t.x, rng) if aug else t.x) for t in trials]
    L = max(len(x) for x in xs)
    X = np.zeros((len(xs), N_FEATURES, L), dtype=np.float32)
    M = np.zeros((len(xs), L), dtype=np.float32)
    for k, x in enumerate(xs):
        X[k, :, :len(x)] = x.T
        M[k, :len(x)] = 1.0
    y = np.array([t.steps for t in trials], dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y)


def count_loss(pred, y):
    """How wrong the predicted count is, on a log scale: |ln(1 + counted) - ln(1 + true)|. Counting 25 when it was 20 costs 0.21;
    counting 1 step when there were none costs 0.69.

    An earlier version used the relative error |counted - true| / (true + 5). It COLLAPSED: on a trial with no steps, one early false step
    cost 12 times as much as missing steps on a walk, so the network learned to count nothing at all and then could not get out of that
    (a softplus that is far below zero has almost no gradient). The log scale keeps the false-step penalty big but bounded."""
    return torch.abs(torch.log1p(pred) - torch.log1p(y)).mean()


def fit(trials, epochs=60, seed=0, lr=3e-3, batch=4, aug=True, width=WIDTH, log=None, max_restarts=2, make_net=None):
    """Train a new network on trials (each has .x and .steps). Returns (network, loss per epoch).
    If a run ends stuck (see is_collapsed: it counts almost none of the steps in its own training walks), it is thrown away and training
    restarts with another seed. The network keeps .restarts (how many times that happened) and .collapsed (True if every attempt got stuck)."""
    for attempt in range(max_restarts + 1):
        net, history = _train_once(trials, epochs, seed + 1000 * attempt, lr, batch, aug, width, log, make_net)
        stuck = is_collapsed(net, trials)
        if not stuck:
            break
        if log:
            log("  training got stuck counting nothing (final loss %.2f): restarting with another seed" % history[-1])
    net.restarts, net.collapsed = attempt, bool(stuck)
    return net, history


def is_collapsed(net, trials):
    """True if the network counts less than COLLAPSE_RATIO of the true steps on the training trials that have steps. Uses only training trials."""
    walks = [t for t in trials if t.steps > 0]
    if not walks:
        return False
    counted = sum(float(density_of(net, t.x).sum()) for t in walks)
    return counted < COLLAPSE_RATIO * sum(t.steps for t in walks)


def _train_once(trials, epochs, seed, lr, batch, aug, width, log, make_net=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net() if make_net else StepNet(width=width)          # make_net: another architecture, for the comparison in arch_compare.py
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(trials))
        total = 0.0
        for i in range(0, len(order), batch):
            chunk = [trials[j] for j in order[i:i + batch]]
            X, M, y = _batch(chunk, rng, aug)
            density = net(X) * M                                  # moments that are only padding count for nothing
            loss = count_loss(density.sum(1), y)
            opt.zero_grad()
            loss.backward()                                       # backpropagation: how much did each weight add to the error?
            opt.step()                                            # every weight moves a little in the direction that shrinks it
            total += loss.item() * len(chunk)
        sched.step()
        history.append(total / len(trials))
        if log and (ep % 20 == 0 or ep == epochs - 1):
            log("  epoch %3d  loss %.3f" % (ep, history[-1]))
    net.eval()
    return net, history


def density_of(net, x):
    """The density curve (steps per reading) of one trial (T, 6), as a NumPy array."""
    net.eval()
    with torch.no_grad():
        f = torch.from_numpy(features(x).T[None].copy())
        return net(f)[0].numpy().astype(np.float64)


def count_of(net, x):
    return float(density_of(net, x).sum())
