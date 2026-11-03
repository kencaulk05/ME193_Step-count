"""The figures: plain, readable, and honest about the scatter."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from steps_data import FS

COLORS = {"walk_slow": "#1f77b4", "walk_normal": "#2ca02c", "walk_fast": "#ff7f0e", "jog": "#d62728", "foot_tap": "#9467bd", "stand": "#7f7f7f", "bike": "#8c564b", "other": "#17becf"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def scatter(true, pred, kinds, path, title, ylabel="steps the network counted"):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    hi = max(max(true), max(pred)) * 1.08 + 1
    ax.fill_between([0, hi], [0, hi * 0.9], [0, hi * 1.1], color="#cccccc", alpha=0.5, label="within 10%")
    ax.plot([0, hi], [0, hi], color="#444444", lw=1)
    for k in sorted(set(kinds)):
        i = [j for j, kk in enumerate(kinds) if kk == k]
        ax.scatter(np.asarray(true)[i], np.asarray(pred)[i], s=34, color=COLORS.get(k, "#333333"), label=k, edgecolor="white", linewidth=0.5)
    ax.set_xlim(-0.04 * hi, hi); ax.set_ylim(-0.04 * hi, hi)                 # a little room below zero, so the trials with no steps (at 0, 0) are not cut in half
    ax.set_xlabel("steps you counted"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11); ax.legend(fontsize=8, frameon=False, loc="upper left")
    _save(fig, path)


def bland_altman(true, pred, path, title):
    true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
    d, m = pred - true, (pred + true) / 2
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(m, d, s=28, color="#1f77b4", edgecolor="white", linewidth=0.5)
    bias, sd = d.mean(), d.std(ddof=1) if len(d) > 1 else 0.0
    ax.axhline(0, color="#444444", lw=1)
    ax.axhline(bias, color="#d62728", lw=1.2, label="average error %+.1f" % bias)
    for s in (-1.96, 1.96):
        ax.axhline(bias + s * sd, color="#d62728", lw=1, ls="--")
    ax.set_xlabel("average of counted and true steps"); ax.set_ylabel("network minus you (steps)")
    ax.set_title(title + "  (dashed: 95%% of errors fall between %+.1f and %+.1f)" % (bias - 1.96 * sd, bias + 1.96 * sd), fontsize=9.5)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


def example(trial, density, path):
    t = np.arange(len(trial.x)) / FS
    am = np.linalg.norm(trial.x[:, :3], axis=1) - 1.0
    gm = np.linalg.norm(trial.x[:, 3:], axis=1)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8.5, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})
    a1.plot(t, gm, color="#ff7f0e", lw=0.8); a1.set_ylabel("rotation\n(degrees/s)")
    a2.plot(t, am, color="#1f77b4", lw=0.8); a2.set_ylabel("acceleration\nminus 1 g (g)")
    a3.fill_between(t, density * FS, color="#2ca02c", alpha=0.35, lw=0); a3.plot(t, density * FS, color="#2ca02c", lw=1)
    a3.set_ylabel("step density\n(steps/s)"); a3.set_xlabel("seconds")
    a3b = a3.twinx(); a3b.plot(t, np.cumsum(density), color="#333333", lw=1.6); a3b.set_ylabel("running count"); a3b.spines["right"].set_visible(True)
    a1.set_title("%s: you counted %d steps, the network counted %.1f" % (trial.name[:48], trial.steps, density.sum()), fontsize=10)
    _save(fig, path)


def loss_curves(histories, path):
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for h in histories:
        ax.plot(h, color="#1f77b4", alpha=0.45, lw=1)
    ax.set_xlabel("epoch (one pass over the training trials)"); ax.set_ylabel("training loss (relative count error)"); ax.set_yscale("log")
    ax.set_title("Training: each line is one of the cross-validation networks", fontsize=10)
    _save(fig, path)


def compare(rows, path):
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    lo = [r[1] - r[2][0] for r in rows]; hi = [r[2][1] - r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(names, vals, yerr=[lo, hi], color=["#999999", "#2ca02c"][:len(rows)], capsize=5)
    for i, v in enumerate(vals):
        ax.text(i, v, "%.1f" % v, ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("average error (steps per trial)"); ax.set_title("Out-of-fold error, with 95% interval", fontsize=10)
    _save(fig, path)


def distance_plot(true_m, pred_m, path, title):
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    hi = max(max(true_m), max(pred_m)) * 1.1
    ax.fill_between([0, hi], [0, hi * 0.9], [0, hi * 1.1], color="#cccccc", alpha=0.5, label="within 10%")
    ax.plot([0, hi], [0, hi], color="#444444", lw=1)
    ax.scatter(true_m, pred_m, s=40, color="#9467bd", edgecolor="white")
    ax.set_xlabel("distance you measured (m)"); ax.set_ylabel("distance estimated (m)"); ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_title(title, fontsize=10); ax.legend(frameon=False, fontsize=9)
    _save(fig, path)
