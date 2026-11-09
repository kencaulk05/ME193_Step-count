#!/usr/bin/env python3
"""Draw the illustrations in docs/images/. Needs numpy, scipy, matplotlib (no PyTorch).

    python docs/make_images.py --trials all_trials.zip       everything
    python docs/make_images.py                               only the drawings that need no recordings

Figures that show a real signal use ONE person's trials only (person A, the first person in the recorder's manifest), and people are letters
everywhere. Figures built from numbers that were recorded in a log rather than computed here say so in their caption (see the README).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from steps_data import FS, Anonymizer, features, load_trials          # noqa: E402
from steps_distance import active_seconds                              # noqa: E402
from steps_live import LiveCounter                                     # noqa: E402
from steps_numpy import NumpyStepNet                                   # noqa: E402

OUT = ROOT / "docs" / "images"
C = {"sig": "#1f77b4", "rot": "#e08a1e", "dens": "#2ca02c", "net": "#6f42c1", "bad": "#d62728", "grey": "#6b7280", "light": "#eef2f7", "ink": "#1d2330", "line": "#aab4c3", "warm": "#fff3e0", "cool": "#e8f1fb", "green": "#e6f4ea", "red": "#fdecea", "purple": "#f0eafa"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name, "(%d KB)" % ((OUT / name).stat().st_size // 1024))


def canvas(w, h, figw=None):
    fig = plt.figure(figsize=(figw or w / 10.0, (figw or w / 10.0) * h / w))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=None, ec=None, fs=10, bold=False, color=None, lw=1.4, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc or C["light"], ec=ec or C["line"], lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or C["ink"], weight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, p, q, text=None, color=None, fs=9, rad=0.0, dy=1.6, lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color or C["grey"], lw=lw, connectionstyle="arc3,rad=%s" % rad, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color=color or C["grey"], linespacing=1.2)


# ---------------------------------------------------------------------------------------------------------------------------------- 1
def system_overview():
    fig, ax = canvas(170, 64, 15)
    box(ax, 2, 24, 20, 16, "Ankle sensor\nMPU-6050", C["cool"], C["sig"], 11, True)
    box(ax, 36, 18, 30, 28, "UNO Q\nmicrocontroller\n\nsketch: reads the sensor\n100 times a second,\ndraws the LED matrix", C["warm"], C["rot"], 9.5)
    ax.add_patch(FancyBboxPatch((82, 3), 56, 58, boxstyle="round,pad=0.4,rounding_size=1.6", fc="#fbfbfd", ec=C["net"], lw=1.8))
    ax.text(110, 57, "UNO Q Linux side: the Python app", ha="center", fontsize=10.5, weight="bold", color=C["net"])
    box(ax, 87, 40, 46, 9, "queue of readings\n(the Bridge call stays light)", C["light"], fs=9)
    box(ax, 87, 26, 46, 9, "live counter\nsliding window, 0.62 s lookahead", C["light"], fs=9)
    box(ax, 87, 9, 46, 12, "the step network\nNumPy copy, 5,857 weights\nsaved in model/step_net.json", C["purple"], C["net"], 9, True)
    arrow(ax, (110, 40), (110, 36)); arrow(ax, (110, 26), (110, 22.5))
    arrow(ax, (22.5, 32), (35.5, 32), "I2C\n100 a second", C["sig"])
    arrow(ax, (66.5, 32), (81.5, 32), "Bridge\nimu(n, ax…gz)", C["rot"])
    box(ax, 146, 42, 22, 17, "Web page\nport 7000\ncount · pace\ndistance · tally", C["green"], C["dens"], 9)
    box(ax, 146, 24, 22, 12, "LED matrix\nthe count\nas a number", C["green"], C["dens"], 9)
    box(ax, 146, 4, 22, 14, "Saved walk\nCSV + manifest\n+ log (recorder\nformat)", C["green"], C["dens"], 8.5)
    arrow(ax, (138.5, 47), (145.5, 50), color=C["dens"]); arrow(ax, (138.5, 33), (145.5, 30), color=C["dens"]); arrow(ax, (138.5, 17), (145.5, 11), color=C["dens"])
    arrow(ax, (146, 55.5), (138.5, 52.5), color=C["grey"], rad=-0.25, ls="--")
    ax.text(142.5, 61.5, "START / STOP,\nthe steps you counted", ha="center", fontsize=8, color=C["grey"], linespacing=1.15)
    ax.text(85, 0.2, "The network is trained on a laptop with PyTorch; the board only runs the saved weights.", fontsize=9, color=C["grey"], style="italic")
    save(fig, "system-overview.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 2
def data_flow(net, trial):
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 150); ax.set_ylim(0, 46); ax.axis("off")
    steps = [("6 numbers\nax ay az\ngx gy gz", C["cool"], C["sig"]), ("8 numbers\n+ total acceleration −1 g\n+ total rotation", C["cool"], C["sig"]), ("5 convolution\nlayers\n16 filters each", C["purple"], C["net"]),
             ("step density\none number per\nreading", C["green"], C["dens"]), ("add it up\nΣ = steps\n(rounded on screen)", C["green"], C["dens"]), ("× step length\n= distance", C["warm"], C["rot"])]
    shapes = ["(T, 6)\n100 a second", "(T, 8)", "(T,)", "(T,)", "one number", "metres"]
    xs = [3 + i * 24.5 for i in range(6)]
    for i, ((txt, fc, ec), x) in enumerate(zip(steps, xs)):
        box(ax, x, 6, 19, 17, txt, fc, ec, 9.5, True)
        ax.text(x + 9.5, 1.8, shapes[i], ha="center", fontsize=8.5, color=C["grey"], linespacing=1.15)
        if i < 5:
            arrow(ax, (x + 19.7, 14.5), (xs[i + 1] - 0.7, 14.5), color=C["grey"])
    x = trial.x
    t = np.arange(len(x)) / FS
    dens = net.density(features(x))
    for idx, ys, col, lab in ((0, np.linalg.norm(x[:, 3:], axis=1), C["rot"], "rotation"), (3, dens * FS, C["dens"], "steps/s")):
        a = fig.add_axes([(xs[idx] + 1) / 150, 29 / 46, 17 / 150, 12 / 46])
        a.plot(t, ys, color=col, lw=1.1); a.set_xticks([]); a.set_yticks([]); a.spines[["left", "bottom"]].set_visible(False)
        a.set_title(lab, fontsize=8.5, color=col, pad=1)
    save(fig, "data-flow.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 3
def signal_anatomy(trial):
    x = trial.x
    t = np.arange(len(x)) / FS
    g = np.linalg.norm(x[:, 3:], axis=1)
    a = np.linalg.norm(x[:, :3], axis=1) - 1.0
    sw, _ = find_peaks(g, height=200, distance=int(0.6 * FS))          # the clear swings only: the small movements at the start and stop are not marked
    imp, _ = find_peaks(np.abs(a), height=0.55, distance=int(0.2 * FS))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(t, g, color=C["rot"], lw=1.2); a1.plot(t[sw], g[sw], "o", color=C["bad"], ms=7)
    for k, i in enumerate(sw):
        a1.annotate("swing %d" % (k + 1), (t[i], g[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=C["bad"])
    a1.set_ylabel("rotation of the ankle\n(degrees/second)"); a1.set_ylim(0, g.max() * 1.25)
    a1.set_title("What one walk looks like at the ankle (%d steps counted, %.0f seconds: %d big swings, plus small movements at the start and stop)" % (trial.steps, trial.seconds, len(sw)), fontsize=11)
    a1.text(0.99, 0.93, "each big swing of the leg with the sensor is one stride = about two steps (both feet)", transform=a1.transAxes, ha="right", fontsize=9.5, color=C["rot"])
    a2.plot(t, a, color=C["sig"], lw=1.0); a2.plot(t[imp], a[imp], "v", color=C["bad"], ms=6)
    a2.set_ylabel("acceleration minus 1 g\n(g)"); a2.set_xlabel("seconds")
    a2.text(0.99, 0.93, "sharp spikes: mostly feet hitting the ground", transform=a2.transAxes, ha="right", fontsize=9.5, color=C["sig"])
    save(fig, "signal-anatomy.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 4
def architecture(net):
    layers = net.layers
    fig, ax = canvas(160, 58, 15)
    ax.text(80, 55, "The step network: %d weights" % net.n_parameters(), ha="center", fontsize=13, weight="bold")
    box(ax, 2, 20, 16, 20, "input\n8 numbers\nevery 10 ms\n(8 × T)", C["cool"], C["sig"], 9.5, True)
    reach, x0 = 0, 24
    prev = (18.7, 30)
    for i, l in enumerate(layers):
        cin, cout, k = l["w"].shape[1], l["w"].shape[0], l["w"].shape[2]
        d = l["dilation"]; reach += (k - 1) // 2 * d
        n = l["w"].size + l["b"].size
        x = x0 + i * 20
        box(ax, x, 17, 16, 26, "conv\nkernel %d\ngap %d\n%d filters\n+ ReLU" % (k, d, cout), C["purple"], C["net"], 9.5, True)
        arrow(ax, prev, (x - 0.6, 30)); prev = (x + 16.7, 30)
        ax.text(x + 8, 12.5, "%d weights" % n, ha="center", fontsize=8.8, color=C["grey"])
        ax.text(x + 8, 8.3, "sees ±%d\nreadings" % reach, ha="center", fontsize=8.8, color=C["net"], linespacing=1.15)
    xh = x0 + 5 * 20
    box(ax, xh, 20, 14, 20, "1×1 conv\n+ softplus\n(never\nnegative)", C["purple"], C["net"], 9, True)
    arrow(ax, prev, (xh - 0.6, 30)); ax.text(xh + 7, 14.5, "%d weights" % (net.head_w.size + net.head_b.size), ha="center", fontsize=8.8, color=C["grey"])
    box(ax, xh + 20, 20, 15, 20, "step density\n1 number per\nreading\n(1 × T)", C["green"], C["dens"], 9.3, True)
    arrow(ax, (xh + 14.7, 30), (xh + 19.4, 30))
    ax.text(xh + 27.5, 14, "Σ = steps", ha="center", fontsize=11, weight="bold", color=C["dens"])
    ax.text(80, 2.2, "Each layer slides the same small filters along time, so a step is recognised wherever it happens. The gaps (1, 2, 4, 8, 16) widen the view to ±%d readings = ±%.2f s." % (reach, reach / FS), ha="center", fontsize=9.5, color=C["grey"], style="italic")
    save(fig, "architecture.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 5
def receptive_field(net):
    half = net.receptive_field // 2
    offs = np.arange(-half, half + 1)
    rows, labels = [], []
    cur = np.array([0])
    rows.append(cur.copy()); labels.append("output: one moment")
    for i, l in reversed(list(enumerate(net.layers))):
        k, d = l["w"].shape[2], l["dilation"]
        taps = np.arange(-(k // 2), k // 2 + 1) * d
        cur = np.unique((cur[:, None] + taps[None, :]).ravel())
        rows.append(cur.copy()); labels.append("layer %d reads these (gap %d)" % (i + 1, d) if i > 0 else "the signal itself, read by layer 1 (gap 1)")
    mat = np.zeros((len(rows), len(offs)))
    for r, cols in enumerate(rows):
        mat[r, cols + half] = 1
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.imshow(mat, aspect="auto", cmap=matplotlib.colors.ListedColormap(["white", C["net"]]), interpolation="nearest", extent=(-half - 0.5, half + 0.5, len(rows) - 0.5, -0.5))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("readings before and after the moment being counted (1 reading = 10 ms)")
    ax.axvline(-half - 0.5, color=C["line"], lw=0.8); ax.axvline(half + 0.5, color=C["line"], lw=0.8)
    for r in range(len(rows) + 1):
        ax.axhline(r - 0.5, color="white", lw=2)
    ax.set_title("Which moments of the signal can change one output number: %d readings = %.2f seconds" % (len(offs), len(offs) / FS), fontsize=11.5)
    ax.spines[["left", "bottom"]].set_visible(False)
    save(fig, "receptive-field.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 6
def count_is_sum(net, trial):
    x = trial.x
    t = np.arange(len(x)) / FS
    d = net.density(features(x))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.fill_between(t, d * FS, color=C["dens"], alpha=0.35, lw=0); ax.plot(t, d * FS, color=C["dens"], lw=1.4)
    ax.set_ylabel("step density (steps per second)", color=C["dens"]); ax.set_xlabel("seconds")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(t, np.cumsum(d), color=C["ink"], lw=2); ax2.set_ylabel("running count = area so far")
    ax2.axhline(trial.steps, color=C["bad"], ls="--", lw=1.3)
    ax2.text(t[-1], trial.steps + 0.3, "you counted %d" % trial.steps, ha="right", color=C["bad"], fontsize=10)
    ax2.text(t[-1], d.sum() - 1.1, "network: %.1f" % d.sum(), ha="right", color=C["ink"], fontsize=10, weight="bold")
    ax2.set_ylim(0, max(trial.steps, d.sum()) * 1.25)
    ax.set_title("The count is the area under the density", fontsize=12)
    fig.text(0.5, -0.03, "This walk was in the data the final network was trained on, so it shows the idea, not the accuracy: the honest numbers are in the results figures.", ha="center", fontsize=8.8, color=C["grey"], style="italic")
    save(fig, "count-is-a-sum.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 7
def training_loop():
    fig, ax = canvas(150, 62, 14)
    ax.text(75, 58.5, "One training step (repeated for every trial, 60 epochs)", ha="center", fontsize=12.5, weight="bold")
    top = [("1  a trial\nthe signal and the\ncount you typed", C["cool"], C["sig"]), ("2  forward pass\nthe network turns the\nsignal into a density", C["purple"], C["net"]),
           ("3  add it up\npredicted steps =\nΣ density", C["green"], C["dens"]), ("4  the loss\n|ln(1 + predicted)\n− ln(1 + yours)|", C["red"], C["bad"])]
    xs = [4, 42, 80, 118]
    for (txt, fc, ec), x in zip(top, xs):
        box(ax, x, 34, 28, 17, txt, fc, ec, 9.5, True)
    for i in range(3):
        arrow(ax, (xs[i] + 28.8, 42.5), (xs[i + 1] - 0.8, 42.5), color=C["grey"])
    box(ax, 118, 6, 28, 17, "5  backward pass\nbackpropagation: how much\ndid each weight add\nto that error?", C["red"], C["bad"], 9, True)
    box(ax, 80, 6, 28, 17, "6  update\nAdam nudges all\n5,857 weights a little,\nthe way that shrinks it", C["purple"], C["net"], 9, True)
    box(ax, 42, 6, 28, 17, "the network is a\nlittle better:\nnext trial", C["light"], C["line"], 9, True)
    arrow(ax, (132, 33), (132, 24.2), color=C["bad"], text="gradient", dy=2.5)
    arrow(ax, (117.2, 14.5), (109, 14.5), color=C["bad"]); arrow(ax, (79.2, 14.5), (71, 14.5), color=C["net"])
    arrow(ax, (42, 14.5), (18, 33.2), color=C["grey"], rad=-0.25, text="again", dy=-5)
    ax.text(75, 1.2, "In PyTorch: loss.backward() does step 5, optimizer.step() does step 6. Nothing says where the steps are: only the total.", ha="center", fontsize=9, color=C["grey"], style="italic")
    save(fig, "training-loop.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 8
def collapse_and_fix():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    labels = ["old start\n0.018 steps per reading", "fixed start\n0.0009 steps per reading"]
    false_steps = [0.018 * 2000, 0.0009 * 2000]
    loss = [np.log(1 + f) for f in false_steps]
    bars = a1.bar(labels, false_steps, color=[C["bad"], C["dens"]], width=0.55)
    for b, f, l in zip(bars, false_steps, loss):
        a1.text(b.get_x() + b.get_width() / 2, f + 1, "%.1f false steps\nloss %.1f" % (f, l), ha="center", fontsize=10)
    a1.set_ylabel("steps counted in a 20-second trial\nwith no steps, before training"); a1.set_ylim(0, 48)
    a1.set_title("Why the old start was a trap", fontsize=11.5)
    a1.text(0.5, -0.27, "A 20 s trial is 2,000 readings. The early updates turned the density down everywhere, and a\nsoftplus far below zero has almost no gradient to climb back out: the network counts nothing.", transform=a1.transAxes, ha="center", fontsize=8.6, color=C["grey"], va="top")
    a2.bar(["old start", "fixed start"], [3.9, 0.6], color=[C["bad"], C["dens"]], width=0.55)
    a2.text(0, 3.9 + 0.12, "2 of 5 runs\ncollapsed (counted\n0% of the true steps)", ha="center", fontsize=9.5)
    a2.text(1, 0.6 + 0.12, "0 of 5 collapsed", ha="center", fontsize=9.5)
    a2.set_ylim(0, 5.6); a2.set_ylabel("average error on held-out trials (steps)")
    a2.set_title("Same data, same recipe, 5 random splits", fontsize=11.5)
    fig.text(0.5, -0.07, "Numbers from the experiment described in the README (5 random splits of the real trials, 30 epochs). The false steps are arithmetic: starting density × 2,000 readings.", ha="center", fontsize=8.6, color=C["grey"], style="italic")
    fig.tight_layout()
    save(fig, "collapse-and-fix.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 9
def live_counter(net, trial):
    x = trial.x
    c = LiveCounter(net, update_every=25)
    live_t, live_c, best = [], [], (-1.0, 0, None, None)
    for i in range(0, len(x) - 24, 25):
        c.push(x[i:i + 25]); live_t.append((i + 25) / FS); live_c.append(c.count)
        if (i + 25) / FS >= 3.0 and float(c.tail.sum()) > best[0]:
            best = (float(c.tail.sum()), i + 25, c.density.copy(), c.tail.copy())       # the moment with the most estimated mass
    _, n_now, fin, tail = best
    full = net.density(features(x))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), gridspec_kw={"height_ratios": [1, 1]})
    tf, tt = np.arange(len(fin)) / FS, (len(fin) + np.arange(len(tail))) / FS
    a1.fill_between(tf, fin * FS, color=C["dens"], alpha=0.5, lw=0, label="final: the whole receptive field has arrived")
    a1.fill_between(tt, tail * FS, color=C["dens"], alpha=0.18, lw=0, hatch="///", ec=C["dens"], label="estimate: nothing to the right of it yet")
    a1.axvline(n_now / FS, color=C["bad"], ls="--"); a1.text(n_now / FS + 0.05, a1.get_ylim()[1] * 0.9, "now", color=C["bad"], fontsize=10)
    a1.annotate("", xy=(len(fin) / FS, 0.3), xytext=(n_now / FS, 0.3), arrowprops=dict(arrowstyle="<->", color=C["bad"]))
    a1.text((len(fin) / FS + n_now / FS) / 2, 0.75, "0.62 s", ha="center", color=C["bad"], fontsize=9)
    a1.set_xlim(0, trial.seconds); a1.set_ylabel("step density\n(steps/second)"); a1.legend(loc="upper left", fontsize=8.8, frameon=False)
    a1.set_title("Counting while you walk: a moment is final once 0.62 s of signal after it has arrived", fontsize=11.5)
    a2.plot(np.arange(len(full)) / FS, np.cumsum(full), color=C["grey"], lw=2, label="the answer for the whole walk (what STOP gives)")
    a2.plot(live_t, live_c, color=C["dens"], lw=2, label="the count on the screen while walking (updated every 0.25 s)")
    a2.set_xlim(0, trial.seconds); a2.set_xlabel("seconds"); a2.set_ylabel("steps so far"); a2.legend(loc="upper left", fontsize=8.8, frameon=False)
    save(fig, "live-counter-timeline.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 10
def results_overview(R):
    n, b = R["network"], R["baseline"]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.6))
    a = axs[0, 0]
    vals = [b["mae"], n["mae"]]; lo = [b["mae"] - R["baseline_mae_ci95"][0], n["mae"] - R["network_mae_ci95"][0]]; hi = [R["baseline_mae_ci95"][1] - b["mae"], R["network_mae_ci95"][1] - n["mae"]]
    a.bar(["classical\ncounter", "network"], vals, yerr=[lo, hi], color=[C["grey"], C["dens"]], capsize=6, width=0.55)
    for i, v in enumerate(vals): a.text(i, v + hi[i] + 0.04, "%.2f" % v, ha="center", va="bottom", fontsize=11, weight="bold")
    a.set_ylabel("steps off, on average"); a.set_title("Average error, 55 trials (95% interval)", fontsize=11); a.set_ylim(0, 1.5)
    a = axs[0, 1]
    w = R["walking_only"]
    a.bar(["classical\ncounter", "network"], [w["baseline"]["mae"], w["network"]["mae"]], color=[C["grey"], C["dens"]], width=0.55)
    for i, v in enumerate([w["baseline"]["mae"], w["network"]["mae"]]): a.text(i, v + 0.02, "%.2f" % v, ha="center", va="bottom", fontsize=11, weight="bold")
    a.set_title("Only the %d trials WITH steps: the classical counter is as good" % w["n"], fontsize=10.5); a.set_ylim(0, 1.5); a.set_ylabel("steps off, on average")
    a = axs[1, 0]
    nz = [r for r in R["oof"] if r["true"] == 0]
    a.bar(["classical\ncounter", "network"], [np.mean([r["baseline"] for r in nz]), np.mean([r["network"] for r in nz])], color=[C["grey"], C["dens"]], width=0.55)
    a.text(0, np.mean([r["baseline"] for r in nz]) + 0.05, "%.1f on average\n(worst: %d steps)" % (np.mean([r["baseline"] for r in nz]), max(r["baseline"] for r in nz)), ha="center", fontsize=9.5)
    a.text(1, 0.05, "0.0\n(worst: %.1f)" % max(r["network"] for r in nz), ha="center", fontsize=9.5)
    a.set_title("Steps counted in the %d trials with NO steps (standing, foot tapping)" % len(nz), fontsize=10.5); a.set_ylim(0, 1.6); a.set_ylabel("steps counted")
    a = axs[1, 1]
    kinds = ["walk_slow", "walk_normal", "walk_fast", "jog", "other", "foot_tap", "stand"]
    names = ["slow", "normal", "fast", "jog", "loops\n(20)", "foot\ntap", "stand"]
    nn = [np.mean([abs(r["network"] - r["true"]) for r in R["oof"] if r["kind"] == k]) for k in kinds]
    bb = [np.mean([abs(r["baseline"] - r["true"]) for r in R["oof"] if r["kind"] == k]) for k in kinds]
    xs = np.arange(len(kinds)); a.bar(xs - 0.2, bb, 0.38, color=C["grey"], label="classical counter"); a.bar(xs + 0.2, nn, 0.38, color=C["dens"], label="network")
    a.set_xticks(xs); a.set_xticklabels(names, fontsize=9); a.legend(frameon=False, fontsize=9); a.set_title("Average error by kind of trial (steps)", fontsize=11); a.set_ylabel("steps off")
    fig.suptitle("Every trial predicted by a model that never saw it", fontsize=12.5, y=1.0)
    fig.tight_layout()
    save(fig, "results-overview.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 11
def architecture_comparison(numbers):
    rows = numbers["variants"]
    fig, ax = plt.subplots(figsize=(10, 4.9))
    ref = rows[0]
    ax.axvspan(ref["lo"], ref["hi"], color=C["dens"], alpha=0.12)
    for i, r in enumerate(rows):
        y = len(rows) - i
        col = C["dens"] if i == 0 else C["net"]
        ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=3, solid_capstyle="round"); ax.plot(r["mean"], y, "o", color=col, ms=9)
        ax.text(1.38, y, "%s restart%s" % (r["restarts"], "" if r["restarts"] == 1 else "s") if r["restarts"] else "", va="center", fontsize=8.5, color=C["bad"])
    ax.set_yticks([len(rows) - i for i in range(len(rows))]); ax.set_yticklabels(["%s  (%s weights)" % (r["what"], format(r["weights"], ",")) for r in rows], fontsize=9.3)
    ax.set_xlim(0.3, 1.5); ax.set_xlabel("average error on trials never seen, steps (dot = average, bar = 95% interval)")
    ax.set_title("Does the architecture matter? Seven variants, same folds, same recipe", fontsize=11.5)
    ax.text(0.31, 0.55, "GRU: not completed (the run was too slow to finish), so there is no number for it.", fontsize=8.8, color=C["grey"], style="italic")
    ax.text(ref["mean"], len(rows) + 0.75, "shipped network's interval", ha="center", fontsize=8.5, color=C["dens"])
    ax.set_ylim(0.3, len(rows) + 1.2)
    save(fig, "architecture-comparison.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 12
def distance_model(net, trials, models, evaluation):
    pts = {}
    for t in trials:
        if t.steps > 0 and np.isfinite(t.distance_m):
            act = active_seconds(net.density(features(t.x)))
            pts.setdefault(t.person, []).append((t.steps / act, t.distance_m / t.steps))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    cols = {"A": C["sig"], "B": C["rot"]}
    allc = np.array([p[0] for v in pts.values() for p in v]); xs = np.linspace(allc.min() - 0.1, allc.max() + 0.1, 20)
    for person, v in sorted(pts.items()):
        v = np.array(v); a1.scatter(v[:, 0], v[:, 1], color=cols.get(person, C["grey"]), s=32, label="person %s: each measured walk" % person, edgecolor="white", lw=0.6)
        m = models["people"].get(person)
        if m: a1.plot(xs, np.clip(m["a"] + m["b"] * xs, 0.3, 1.2), color=cols.get(person, C["grey"]), lw=2)
    a1.plot(xs, np.clip(models["a"] + models["b"] * xs, 0.3, 1.2), color=C["grey"], lw=2, ls="--", label="one model for everyone")
    a1.set_xlabel("cadence while stepping (steps per second)"); a1.set_ylabel("step length (metres per step)"); a1.legend(frameon=False, fontsize=8.8)
    a1.set_title("Step length changes with speed, and between people", fontsize=11)
    e = evaluation
    names = ["one model\nfor both"] + ["person %s,\nown walks" % k for k in sorted(e["distance_per_person_mape"])]
    vals = [e["distance_end_to_end_mape"]] + [e["distance_per_person_mape"][k] for k in sorted(e["distance_per_person_mape"])]
    a2.bar(names, vals, color=[C["grey"]] + [cols.get(k, C["grey"]) for k in sorted(e["distance_per_person_mape"])], width=0.6)
    for i, v in enumerate(vals): a2.text(i, v + 0.3, "%.0f%%" % v, ha="center", fontsize=11, weight="bold")
    a2.set_ylabel("distance error, end to end (%)"); a2.set_ylim(0, max(vals) * 1.3); a2.set_title("Distance = steps × step length", fontsize=11)
    fig.text(0.5, -0.04, "Walks of 4.8 to 5.8 m only, each predicted by a model that never saw it. One step out on a 10-step walk is already about 10%.", ha="center", fontsize=8.8, color=C["grey"], style="italic")
    fig.tight_layout()
    save(fig, "distance-model.png")


# ---------------------------------------------------------------------------------------------------------------------------------- 13
def led_digits():
    src = (ROOT / "apps" / "gait-steps-live" / "sketch" / "sketch.ino").read_text(encoding="utf-8")
    body = re.search(r"DIGITS\[10\]\[7\] = \{(.*?)\};", src, re.S).group(1)
    font = [[int(h, 16) for h in re.findall(r"0x([0-9A-Fa-f]{2})", row)] for row in re.findall(r"\{([^{}]*)\}", body)]
    assert len(font) == 10 and all(len(r) == 7 for r in font)

    def frame(n):
        f = np.zeros((8, 13), dtype=int)
        over = n > 99; n = min(max(n, 0), 99)

        def digit(d, c0):
            for r in range(7):
                for c in range(5):
                    if font[d][r] & (1 << (4 - c)): f[r, c0 + c] = 1
        if n >= 10: digit(n // 10, 1); digit(n % 10, 7)
        else: digit(n, 4)
        if over: f[7, :] = 1
        return f
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.6))
    for ax, (n, cap) in zip(axs, ((7, "7 steps"), (42, "42 steps"), (99, "99 steps"), (150, "99 or more:\nline under the 99"))):
        f = frame(n)
        for r in range(8):
            for c in range(13):
                ax.add_patch(Circle((c, 7 - r), 0.4, color=C["rot"] if f[r, c] else "#d7dde6"))
        ax.set_xlim(-0.8, 12.8); ax.set_ylim(-0.8, 7.8); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(cap, fontsize=10)
    fig.suptitle("The LED matrix shows the step count as a number (13 × 8 lights, a 5 × 7 font)", fontsize=11.5, y=1.04)
    save(fig, "led-matrix.png")
    return {n: int(frame(n).sum()) for n in (7, 42, 99, 150)}


# ---------------------------------------------------------------------------------------------------------------------------------- 14
def decisions():
    rows = [("Sensors", "JY901 over a serial port", "never answered, even when asked directly", "dropped: ankle MPU-6050 only", "x"),
            ("Second sensor", "a GY-9250 on pins A4 and A5", "did not respond in the test app", "dropped: one sensor", "x"),
            ("Labels", "only a count per trial", "no marks for where each step is", "network outputs a step density; the sum is the count", "ok"),
            ("First detector", "peaks, threshold relative to each recording", "counted over 100 strides on a still sensor", "absolute thresholds, in g and degrees/s", "fix"),
            ("Loss", "relative error", "network collapsed to counting nothing", "log-scale loss", "fix"),
            ("Training start", "density 0.018 per reading", "2 of 5 runs still collapsed", "start at 0.0009, restart a stuck run", "fix"),
            ("Stuck-run test", "training loss above 0.4", "flagged a slow but healthy run", "count under 30% of own training steps", "fix"),
            ("Architecture", "reasoned, not tuned", "7 variants within noise (GRU unfinished)", "keep the shipped network", "ok"),
            ("What the screen shows", "the fractional count", "whole steps match whole-step truth better", "rounded count (0.62 against 0.67 steps off)", "ok"),
            ("Names", "names in results and files", "the repo is public", "people are letters everywhere", "ok")]
    H = 12 * len(rows) + 14
    fig, ax = canvas(190, H, 15)
    cols = [(3, "Topic"), (28, "What we tried"), (80, "What happened"), (128, "Decision")]
    for x, h in cols:
        ax.text(x, H - 4, h, fontsize=11, weight="bold", color=C["ink"], va="center")
    sty = {"ok": (C["green"], C["dens"], "kept"), "fix": (C["warm"], C["rot"], "fixed"), "x": (C["red"], C["bad"], "dropped")}
    for i, (topic, tried, happened, decision, st) in enumerate(rows):
        y = H - 14 - 12 * i
        fc, ec, tag = sty[st]
        ax.add_patch(FancyBboxPatch((1, y), 186, 9.5, boxstyle="round,pad=0.3,rounding_size=1", fc=fc, ec=ec, lw=1.1))
        ax.text(3, y + 4.75, topic, fontsize=9.8, weight="bold", va="center")
        ax.text(28, y + 4.75, tried, fontsize=9.3, va="center"); ax.text(80, y + 4.75, happened, fontsize=9.3, va="center"); ax.text(128, y + 4.75, decision, fontsize=9.3, va="center")
        ax.text(185, y + 4.75, tag, fontsize=8.5, ha="right", va="center", color=ec, weight="bold")
    save(fig, "decisions.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", default=None)
    a = ap.parse_args()
    net = NumpyStepNet.load(ROOT / "models" / "step_net.json")
    R = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    models = json.loads((ROOT / "models" / "distance.json").read_text(encoding="utf-8"))
    ev = json.loads((ROOT / "models" / "evaluation.json").read_text(encoding="utf-8"))
    numbers = json.loads((ROOT / "docs" / "architecture_comparison_numbers.json").read_text(encoding="utf-8"))
    print("drawings that need no recordings:")
    system_overview(); architecture(net); receptive_field(net); training_loop(); collapse_and_fix(); results_overview(R); architecture_comparison(numbers); decisions(); lit = led_digits()
    print("  led pixel counts:", lit)
    if a.trials and Path(a.trials).exists():
        print("drawings that show a real signal (person A only):")
        trials, _ = load_trials(a.trials, anonymizer=Anonymizer())
        walk = next(t for t in trials if t.person == "A" and t.kind == "walk_normal" and t.steps == 11)
        data_flow(net, walk); signal_anatomy(walk); count_is_sum(net, walk); live_counter(net, walk); distance_model(net, trials, models, ev)
    else:
        print("(no --trials: the five figures that show a real signal were not redrawn)")


if __name__ == "__main__":
    main()
