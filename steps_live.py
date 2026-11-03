"""Counting steps while you walk.

The network looks 0.62 s ahead and 0.62 s behind each moment (its receptive field is 125 readings), so the density at a moment can only
be FINAL once 0.62 s of signal after it has arrived. The live counter therefore keeps two numbers:

    committed    the sum of the density over moments that are final. It only ever goes up, and it is exact.
    provisional  committed + the density of the last 0.62 s as the network sees it so far (with nothing yet to the right).
                 This is what the screen shows while you walk; it can change by a little as the next readings arrive.

When the readings stop, finish() makes the last moments final, and the count is then EXACTLY the offline count of the whole trial.
"""
import numpy as np

from steps_data import FS, features
from steps_numpy import NumpyStepNet


class LiveCounter:
    def __init__(self, net: NumpyStepNet, update_every=25):
        self.net = net
        self.half = net.receptive_field // 2           # 62 readings: how far the network looks each way
        self.update_every = update_every               # recompute every 0.25 s
        self.reset()

    def reset(self):
        self.x = np.zeros((0, 6), dtype=np.float32)    # every reading so far (a walk is minutes, so this stays small)
        self.n = 0
        self.committed_upto = 0                        # moments before this one are final
        self.committed = 0.0
        self.tail = np.zeros(0)                        # the provisional density of the not-yet-final moments
        self._since = 0
        self.finished = False
        self.density = np.zeros(0)                     # final density so far (for drawing)

    # ---- readings in ----
    def push(self, readings):
        """readings: (k, 6) array, or one reading of 6 numbers."""
        if self.finished:
            raise RuntimeError("this walk is finished: call reset() to start another")
        r = np.asarray(readings, dtype=np.float32).reshape(-1, 6)
        self.x = np.concatenate([self.x, r])
        self.n += len(r)
        self._since += len(r)
        if self._since >= self.update_every:
            self._update()
        return self.count

    def _window(self):
        start = max(0, self.committed_upto - self.half)
        return start, self.net.density(features(self.x[start:]))

    def _update(self):
        self._since = 0
        final_to = self.n - self.half                                # moments before this have their whole receptive field
        start, d = self._window()
        if final_to > self.committed_upto:
            new = d[self.committed_upto - start:final_to - start]
            self.committed += float(new.sum())
            self.density = np.concatenate([self.density, new])
            self.committed_upto = final_to
        self.tail = d[max(self.committed_upto - start, 0):]

    def finish(self):
        """No more readings: everything becomes final, and the count equals the offline count of the whole walk."""
        if self.finished:
            return self.committed
        if self.n:
            start, d = self._window()
            new = d[self.committed_upto - start:]
            self.committed += float(new.sum())
            self.density = np.concatenate([self.density, new])
            self.committed_upto = self.n
        self.tail = np.zeros(0)
        self.finished = True
        return self.committed

    # ---- numbers out ----
    @property
    def count(self):
        """Steps so far: exact up to 0.62 s ago, estimated for the last 0.62 s."""
        return self.committed + float(self.tail.sum())

    def steps_per_minute(self, seconds=10.0):
        """Recent pace, from the final density of the last few seconds."""
        k = int(seconds * FS)
        d = self.density[-k:] if len(self.density) else np.zeros(0)
        return 60.0 * float(d.sum()) / max(len(d) / FS, 1e-9) if len(d) >= FS else 0.0

    def active_seconds(self):
        from steps_distance import active_seconds
        d = np.concatenate([self.density, self.tail])
        return active_seconds(d) if len(d) else 0.0
