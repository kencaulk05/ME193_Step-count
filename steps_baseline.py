"""The classical step counter the network has to beat: smooth a signal, count its peaks, with ABSOLUTE thresholds.

Two families, because it is not obvious which signal counts steps best:
    gyro   one peak in the total rotation per leg swing of the ankle with the sensor; steps = 2 x peaks (both feet, if they alternate)
    acc    one peak in the band-passed total acceleration per heel strike
A threshold relative to each recording's own range counts noise as steps on a motionless sensor, so these thresholds are in g and degrees/s.
The best settings are found on the TRAINING trials only (the smallest average error), then used unchanged on the test trials.
"""
import itertools

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from steps_data import FS


def _lp(x, hz):
    return sosfiltfilt(butter(3, hz, btype="low", fs=FS, output="sos"), x)


def _bp(x, lo, hi):
    return sosfiltfilt(butter(3, [lo, hi], btype="band", fs=FS, output="sos"), x)


def count_steps(x, family, threshold, min_gap_s):
    """Steps counted in one trial (T, 6)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 40:
        return 0.0
    if family == "gyro":
        s = _lp(np.linalg.norm(x[:, 3:], axis=1), 3.0)                 # degrees/s
        peaks, _ = find_peaks(s, height=threshold, distance=max(1, int(min_gap_s * FS)))
        return 2.0 * len(peaks)
    if family == "acc":
        a = np.linalg.norm(x[:, :3], axis=1)
        s = _lp(np.abs(_bp(a - a.mean(), 1.0, 8.0)), 4.0)               # g
        peaks, _ = find_peaks(s, height=threshold, distance=max(1, int(min_gap_s * FS)))
        return float(len(peaks))
    raise ValueError(family)


GRID = {"gyro": ([30, 50, 70, 100, 140, 190], [0.45, 0.6, 0.75, 0.9]), "acc": ([0.05, 0.08, 0.12, 0.18, 0.26, 0.4], [0.25, 0.32, 0.4, 0.5])}


class Baseline:
    """Fit the best family, threshold and minimum gap on training trials; predict on others."""

    def fit(self, trials):
        best = None
        for fam, (ths, gaps) in GRID.items():
            for th, gap in itertools.product(ths, gaps):
                err = np.mean([abs(count_steps(t.x, fam, th, gap) - t.steps) for t in trials])
                if best is None or err < best[0]:
                    best = (err, fam, th, gap)
        self.train_mae, self.family, self.threshold, self.min_gap_s = best
        return self

    def predict(self, trial):
        return count_steps(trial.x, self.family, self.threshold, self.min_gap_s)

    def describe(self):
        unit = "degrees/s" if self.family == "gyro" else "g"
        return "%s peaks above %g %s at least %g s apart" % (self.family, self.threshold, unit, self.min_gap_s)
