"""Synthetic ankle trials, ONLY to test the machinery (loading, training, evaluating, exporting) with a known right answer.
They are not evidence about real accuracy: the real numbers come from the trials you record. The generator is deliberately crude:
heel strikes of both feet are sharp acceleration impulses (the sensor foot's stronger), the leg with the sensor swings once per stride,
and the hard negatives (a tapping foot, standing and shifting, air-biking) move without sharp impacts."""
import numpy as np

from steps_data import FS, Trial


def _bell(t, centre, sigma):
    return np.exp(-0.5 * ((t - centre) / sigma) ** 2)


def make_trial(rng, kind, steps=None, person="synth", cadence=None, name=None):
    lead, tail = rng.uniform(2.0, 4.0), rng.uniform(2.0, 4.0)
    if kind in ("walk_slow", "walk_normal", "walk_fast", "jog"):
        lo, hi = {"walk_slow": (1.2, 1.7), "walk_normal": (1.6, 2.2), "walk_fast": (2.0, 2.7), "jog": (2.5, 3.2)}[kind]
        c = cadence if cadence is not None else rng.uniform(lo, hi)
        n = int(steps if steps is not None else rng.integers(12, 41))
        dur = lead + n / c + tail
    else:
        c, n = 0.0, 0
        dur = rng.uniform(15.0, 30.0)
    T = int(dur * FS)
    t = np.arange(T) / FS
    x = np.zeros((T, 6), dtype=np.float64)
    x[:, 0] = -0.95 + rng.normal(0, 0.01)                                              # gravity runs down the shin
    x[:, :3] += rng.normal(0, 0.015, (T, 3))
    x[:, 3:] += rng.normal(0, 1.0, (T, 3))
    if n:
        step_len = 0.30 + 0.17 * c
        for i in range(n):
            ti = lead + (i + 0.5) / c * (1 + rng.normal(0, 0.03))
            mine = (i % 2 == 0)                                                        # a heel strike of the foot with the sensor
            amp = rng.uniform(0.8, 2.5) * (1.0 if mine else 0.35) * (1.3 if kind == "jog" else 1.0)
            tt = t - ti
            imp = np.where(tt >= 0, np.exp(-np.maximum(tt, 0) / 0.04) * np.sin(2 * np.pi * 18 * tt), 0.0) * amp
            x[:, 0] += imp * 0.8; x[:, 1] += imp * 0.5; x[:, 2] += imp * 0.3
            if mine:
                g = rng.uniform(120, 350) * (1.2 if kind == "jog" else 1.0)
                x[:, 4] += g * _bell(t, ti - 0.25, 0.12); x[:, 5] += 0.4 * g * _bell(t, ti - 0.25, 0.12)
    elif kind == "foot_tap":
        f = rng.uniform(1.5, 3.0)
        env = (np.sin(2 * np.pi * rng.uniform(0.05, 0.12) * t) > -0.3).astype(float)
        x[:, 4] += rng.uniform(25, 70) * np.sin(2 * np.pi * f * t) * env
        x[:, 2] += rng.uniform(0.05, 0.2) * np.abs(np.sin(np.pi * f * t)) * env
    elif kind == "bike":
        f = rng.uniform(0.8, 1.2)
        x[:, 4] += rng.uniform(100, 200) * np.sin(2 * np.pi * f * t); x[:, 5] += rng.uniform(60, 140) * np.cos(2 * np.pi * f * t)
        x[:, 1] += rng.uniform(0.2, 0.5) * np.sin(2 * np.pi * f * t); x[:, 2] += rng.uniform(0.2, 0.5) * np.cos(2 * np.pi * f * t)
    else:                                                                              # stand: slow weight shifts
        x[:, 3:] += rng.uniform(0, 8) * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t)[:, None] * rng.normal(0, 1, 3)
        x[:, :3] += 0.03 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t)[:, None]
    dist = n * (0.30 + 0.17 * c) * (1 + rng.normal(0, 0.03)) if n else float("nan")
    return Trial(name=name or "%s_%s" % (kind, rng.integers(1_000_000)), person=person, kind=kind, steps=n, distance_m=dist, x=x.astype(np.float32), source="synthetic")


def make_set(seed, n_walk=36, n_neg=14, person="synth"):
    rng = np.random.default_rng(seed)
    kinds = ["walk_slow", "walk_normal", "walk_fast", "jog"]
    out = [make_trial(rng, kinds[i % 4], person=person, name="synth_walk_%d" % i) for i in range(n_walk)]
    out += [make_trial(rng, ["foot_tap", "stand", "bike"][i % 3], person=person, name="synth_neg_%d" % i) for i in range(n_neg)]
    return out
