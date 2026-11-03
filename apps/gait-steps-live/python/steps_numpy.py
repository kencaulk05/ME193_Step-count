"""The step network in plain NumPy: the same arithmetic as the PyTorch model, with no PyTorch needed (the UNO Q runs this one).

The network is five layers of 1-D convolution, each looking at a few neighbouring moments of the sensor signal, then a last layer that
turns what they found into one number per moment: how many steps are happening RIGHT NOW (steps per reading, never negative).
Add those numbers up over a trial and you have the step count.
"""
import json

import numpy as np


def conv_same(x, w, b, dilation):
    """1-D convolution (cross-correlation, as in PyTorch) with zero padding so the output has as many moments as the input.
    x: (T, Cin). w: (Cout, Cin, K), K odd. b: (Cout,). Returns (T, Cout)."""
    T = x.shape[0]
    K = w.shape[2]
    pad = dilation * (K - 1) // 2
    xp = np.pad(x, ((pad, pad), (0, 0)))
    out = np.zeros((T, w.shape[0]), dtype=np.float64) + b
    for k in range(K):
        out += xp[k * dilation:k * dilation + T] @ w[:, :, k].T
    return out


class NumpyStepNet:
    def __init__(self, layers, head_w, head_b, meta=None):
        self.layers = [{"w": np.asarray(l["w"], dtype=np.float64), "b": np.asarray(l["b"], dtype=np.float64), "dilation": int(l["dilation"])} for l in layers]
        self.head_w = np.asarray(head_w, dtype=np.float64)
        self.head_b = np.asarray(head_b, dtype=np.float64)
        self.meta = meta or {}

    @property
    def receptive_field(self):
        return 1 + sum(l["dilation"] * (l["w"].shape[2] - 1) for l in self.layers)

    def density(self, feats):
        """(T, 8) features -> (T,) steps per reading."""
        h = np.asarray(feats, dtype=np.float64)
        for l in self.layers:
            h = np.maximum(conv_same(h, l["w"], l["b"], l["dilation"]), 0.0)          # convolution, then ReLU
        z = conv_same(h, self.head_w, self.head_b, 1)[:, 0]
        return np.logaddexp(0.0, z)                                                    # softplus: never negative

    def count(self, feats):
        return float(self.density(feats).sum())

    # ---- saving and loading (a small JSON file: about 6000 numbers) ----
    def to_dict(self):
        return {"layers": [{"w": l["w"].tolist(), "b": l["b"].tolist(), "dilation": l["dilation"]} for l in self.layers],
                "head_w": self.head_w.tolist(), "head_b": self.head_b.tolist(), "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        return cls(d["layers"], d["head_w"], d["head_b"], d.get("meta"))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def n_parameters(self):
        return sum(l["w"].size + l["b"].size for l in self.layers) + self.head_w.size + self.head_b.size
