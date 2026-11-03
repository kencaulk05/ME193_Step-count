#!/usr/bin/env python3
"""Does the choice of network architecture matter? A sensitivity check, not a search for a winner.

    python arch_compare.py --trials all_trials.zip                    all variants (about 40 minutes)
    python arch_compare.py --trials all_trials.zip --variants narrow wide

Every variant is trained and tested EXACTLY like the shipped network: the same 5 cross-validation folds (so the comparison is paired), the same
60 epochs, loss, augmentation, starting density, restart guard and extra negative recordings. Only the architecture changes. (The GRU is the one
exception in training: it is fed one trial at a time, because padding a batch would change what a bidirectional recurrent network sees.)

THE RULE, FIXED BEFORE ANY RESULT WAS SEEN: the shipped architecture stays unless another variant's average error is lower by at least 0.15 steps
AND the 95% interval of the paired difference lies entirely below zero. Otherwise every variant is just a report of how much the answer depends on
these choices. With a few dozen trials the intervals are wide, so most differences cannot be told apart from noise, and picking the best-looking
variant on the same trials would bias its result upward.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import steps_eval as ev
from steps_data import FS, load_legacy_negatives, load_trials

RULE_MARGIN = 0.15


def paired_ci(true, a, b, n=4000, seed=0):
    """95% interval of mean|a - true| - mean|b - true| over resampled trials. Negative means a is better than b."""
    true, a, b = (np.asarray(v, dtype=float) for v in (true, a, b))
    d = np.abs(a - true) - np.abs(b - true)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(args):
    import steps_model as sm
    import steps_alt as alt
    if args.synthetic:
        from synth import make_set
        trials, legacy = make_set(args.seed, n_walk=20, n_neg=10), []
    else:
        trials, _ = load_trials(args.trials)
        legacy = load_legacy_negatives(args.legacy) if args.legacy and Path(args.legacy).is_dir() else []
    true = np.array([t.steps for t in trials], dtype=float)
    k = max(2, min(args.folds, len(trials) // 3))
    folds = ev.make_folds(trials, k, args.seed)
    names = args.variants or list(alt.VARIANTS)
    if "reference" not in names:
        names = ["reference"] + names
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    saved = out / "arch_compare.json"
    if args.resume and saved.exists():                               # a long run that was interrupted: keep the variants that finished
        old = json.loads(saved.read_text(encoding="utf-8"))
        if old.get("epochs") == args.epochs and old.get("folds") == k and np.allclose(old.get("true", []), true):
            results = {n: v for n, v in old["variants"].items() if n in names}
            print("  resuming: already done: " + ", ".join(results), flush=True)
    for name in names:
        if name in results:
            continue
        what, build, batch = alt.VARIANTS[name]
        t0 = time.time()
        pred = np.full(len(trials), np.nan)
        restarts = []
        n_params = sum(p.numel() for p in build().parameters())
        for fold in folds:
            test = set(fold)
            net, _ = sm.fit([t for i, t in enumerate(trials) if i not in test] + legacy, epochs=args.epochs, seed=args.seed, batch=batch, make_net=build)
            restarts.append(net.restarts)
            for i in fold:
                pred[i] = sm.count_of(net, trials[i].x)
        m = ev.metrics(true, pred)
        walk = true > 0
        results[name] = {"what": what, "params": int(n_params), "restarts": restarts, "seconds": round(time.time() - t0), "pred": pred.tolist(), "metrics": m,
                         "mae_ci95": ev.bootstrap_ci(true, pred, ev.mae, seed=args.seed), "mae_walking": float(np.mean(np.abs(pred[walk] - true[walk]))) if walk.any() else float("nan")}
        print("  %-10s %6d weights  average error %.2f steps (%.2f to %.2f)  walking-only %.2f  false steps on no-step trials %.2f  restarts %d  (%d s)" % (
            name, n_params, m["mae"], *results[name]["mae_ci95"], results[name]["mae_walking"], m["false_steps_mean"], sum(1 for r in restarts if r), results[name]["seconds"]), flush=True)
        (out / "arch_compare.json").write_text(json.dumps({"true": true.tolist(), "variants": results, "epochs": args.epochs, "folds": k}, indent=1, default=float), encoding="utf-8")
    ref = results["reference"]["pred"]
    for name, r in results.items():
        r["vs_reference"] = None if name == "reference" else dict(zip(("mean", "lo", "hi"), paired_ci(true, r["pred"], ref, seed=args.seed)))
        v = r["vs_reference"]
        r["beats_reference_by_rule"] = bool(v and v["hi"] < 0 and -v["mean"] >= RULE_MARGIN)
    (out / "arch_compare.json").write_text(json.dumps({"true": true.tolist(), "variants": results, "epochs": args.epochs, "folds": k}, indent=1, default=float), encoding="utf-8")
    write_summary(out / "architecture_comparison.md", results, len(trials), k, args)
    return results


def write_summary(path, results, n_trials, k, args):
    L = ["# Does the architecture matter?\n"]
    if args.synthetic:
        L.append("**Made-up trials: this checks the code, not the answer.**\n")
    L.append("%d trials, %d-fold cross-validation, the same folds for every variant, %d epochs. Only the architecture changes. Errors are the average number of steps off on trials the network never saw (95%% interval from resampling the trials).\n" % (n_trials, k, args.epochs))
    L.append("**The rule, fixed before any result was seen:** the shipped network stays unless another variant is better by at least %.2f steps with the paired 95%% interval entirely below zero.\n" % RULE_MARGIN)
    L.append("| Variant | What it is | Weights | Average error, steps (95% interval) | Trials with steps only | Difference from the shipped network (95% interval) | Restarts |\n|---|---|---|---|---|---|---|")
    for name, r in results.items():
        v = r["vs_reference"]
        diff = "reference" if v is None else "%+.2f (%+.2f to %+.2f)%s" % (v["mean"], v["lo"], v["hi"], "  **beats it by the rule**" if r["beats_reference_by_rule"] else "")
        L.append("| %s | %s | %d | %.2f (%.2f to %.2f) | %.2f | %s | %d of %d |" % (name, r["what"], r["params"], r["metrics"]["mae"], *r["mae_ci95"], r["mae_walking"], diff, sum(1 for x in r["restarts"] if x), len(r["restarts"])))
    winners = [n for n, r in results.items() if r["beats_reference_by_rule"]]
    L.append("\n## What this says\n")
    if winners:
        L.append("By the rule, %s beat the shipped network." % ", ".join(winners))
    else:
        L.append("**No variant beat the shipped network by the rule**, so it stays. Where a variant looks a little better or worse, the paired interval includes zero: with this few trials the difference cannot be told apart from noise.")
    L.append("\nA negative difference means the variant makes fewer mistakes than the shipped network. The intervals show how much each number would move with a different set of trials, not how the network would do on a new person or a new kind of walking.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--legacy", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="*", help="which variants (the shipped one is always included)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run: skip the variants already in arch_compare.json")
    a = ap.parse_args()
    if not a.trials and not a.synthetic:
        ap.error("give --trials all_trials.zip")
    run(a)
