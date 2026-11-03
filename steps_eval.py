"""Honest evaluation: every trial is predicted by a model that NEVER saw it, and the errors come with an uncertainty."""
import numpy as np


def make_folds(trials, k, seed=0):
    """k groups of trial indices, each kind of trial spread across the groups (so every group tests walking AND tapping AND standing)."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    by_kind = {}
    for i, t in enumerate(trials):
        by_kind.setdefault(t.kind, []).append(i)
    shift = int(rng.integers(k))
    for kind in sorted(by_kind):
        for n, i in enumerate(rng.permutation(by_kind[kind])):
            folds[(n + shift) % k].append(int(i))
        shift += len(by_kind[kind])
    return [f for f in folds if f]


def cross_validate(trials, train_fn, predict_fn, k=5, seed=0):
    """train_fn(training trials) -> model; predict_fn(model, trial) -> value. Returns the out-of-fold predictions, aligned with trials."""
    pred = np.full(len(trials), np.nan)
    for fold in make_folds(trials, k, seed):
        test = set(fold)
        model = train_fn([t for i, t in enumerate(trials) if i not in test])
        for i in fold:
            pred[i] = predict_fn(model, trials[i])
    return pred


def metrics(true, pred):
    true, pred = np.asarray(true, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    err = pred - true
    out = {"n": int(len(true)), "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)), "rmse": float(np.sqrt(np.mean(err ** 2))),
           "sd": float(np.std(err, ddof=1)) if len(err) > 1 else 0.0, "worst": float(np.max(np.abs(err))) if len(err) else 0.0,
           "within1": float(np.mean(np.abs(err) <= 1.0)), "within2": float(np.mean(np.abs(err) <= 2.0))}
    pos = true > 0
    out["mape"] = float(100.0 * np.mean(np.abs(err[pos]) / true[pos])) if pos.any() else float("nan")
    out["n_zero"] = int((~pos).sum())
    out["false_steps_mean"] = float(np.mean(pred[~pos])) if (~pos).any() else float("nan")
    out["false_steps_max"] = float(np.max(pred[~pos])) if (~pos).any() else float("nan")
    return out


def bootstrap_ci(true, pred, stat, n=2000, seed=0):
    """95% interval of stat(true, pred) from resampling the trials: how much the number would move with a different set of trials."""
    true, pred = np.asarray(true, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals = [stat(true[i], pred[i]) for i in (rng.integers(0, len(true), len(true)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mae(t, p):
    return float(np.mean(np.abs(p - t)))


def mape(t, p):
    pos = t > 0
    return float(100.0 * np.mean(np.abs(p[pos] - t[pos]) / t[pos])) if pos.any() else float("nan")


def by_kind(trials, pred):
    rows = {}
    for t, p in zip(trials, pred):
        rows.setdefault(t.kind, []).append((t.steps, p))
    return {k: metrics([a for a, _ in v], [b for _, b in v]) for k, v in sorted(rows.items())}
