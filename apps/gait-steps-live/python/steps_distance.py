"""How far did you walk? Steps times the length of a step, and the length of a step depends on how fast you walk (longer when faster).

    step length = a + b x cadence        (cadence = steps per second while walking)
fitted on the trials where you measured the course. With too few of them it falls back to one constant step length. The honest number to
report is the leave-one-out error: each measured walk is predicted by a model that never saw it.
"""
import numpy as np

from steps_data import FS


def active_seconds(density):
    """How long the person was really stepping: from when 5% of the steps had happened to when 95% had."""
    d = np.asarray(density, dtype=np.float64)
    c = np.cumsum(d)
    if c[-1] < 1.0:
        return max(len(d) / FS, 1e-6)
    lo, hi = np.searchsorted(c, 0.05 * c[-1]), np.searchsorted(c, 0.95 * c[-1])
    return max((hi - lo) / FS, 1.0)


class DistanceModel:
    MIN_FOR_SLOPE = 6

    def fit(self, steps, active_s, distance_m):
        steps, active_s, distance_m = (np.asarray(v, dtype=np.float64) for v in (steps, active_s, distance_m))
        step_len = distance_m / np.maximum(steps, 1)
        cad = steps / np.maximum(active_s, 1e-6)
        self.constant = float(np.median(step_len))
        if len(steps) >= self.MIN_FOR_SLOPE and np.ptp(cad) > 0.3:
            b, a = np.polyfit(cad, step_len, 1)
            self.a, self.b, self.kind = float(a), float(b), "step length = %.3f + %.3f x cadence" % (a, b)
        else:
            self.a, self.b, self.kind = self.constant, 0.0, "constant step length %.3f m" % self.constant
        return self

    def step_length(self, steps, active_s):
        cad = float(steps) / max(float(active_s), 1e-6)
        return min(max(self.a + self.b * cad, 0.3), 1.2)

    def predict(self, steps, active_s):
        return float(steps) * self.step_length(steps, active_s)


def leave_one_out(steps, active_s, distance_m):
    """Percentage error of each measured walk when predicted by a model fitted on the others. Returns (errors in %, model description)."""
    steps, active_s, distance_m = (np.asarray(v, dtype=np.float64) for v in (steps, active_s, distance_m))
    errs = []
    for i in range(len(steps)):
        keep = np.arange(len(steps)) != i
        m = DistanceModel().fit(steps[keep], active_s[keep], distance_m[keep]) if keep.sum() >= 2 else DistanceModel().fit(steps, active_s, distance_m)
        errs.append(100.0 * (m.predict(steps[i], active_s[i]) - distance_m[i]) / distance_m[i])
    return np.array(errs)
