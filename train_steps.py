#!/usr/bin/env python3
"""Train the step counter and measure how good it is.

    python train_steps.py --trials all_trials.zip           the zip from the step recorder's "Prepare zip of everything" button
    python train_steps.py --trials all_trials.zip --quick   a faster, rougher run to check that everything works

What it does, in order:
  1. loads your trials (and the older standing and air-biking recordings, which have no steps, as extra negative examples)
  2. CROSS-VALIDATION: every trial is predicted by a network that never saw it. Same for the classical peak-counting baseline.
  3. the same test on a NEW PERSON, when there are two people with enough trials
  4. how far you walked: steps x step length, tested the same honest way
  5. trains the final network on everything, and writes models/ and results/ (figures, results.json and summary.md)
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

import steps_eval as ev
import steps_export as export
import steps_plots as plots
from steps_baseline import Baseline
from steps_data import FS, load_legacy_negatives, load_trials
from steps_distance import DistanceModel, active_seconds, leave_one_out


def fmt(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else ("%." + str(d) + "f") % x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", help="zip or folder from the step recorder")
    ap.add_argument("--synthetic", action="store_true", help="use made-up trials (to test that the machinery works; NOT real accuracy)")
    ap.add_argument("--legacy", default=None, help="folder with the older gait recordings (standing, air-biking): extra negatives")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="30 epochs, 3 folds")
    a = ap.parse_args(argv)
    if a.quick:
        a.epochs, a.folds = 30, 3
    if not a.trials and not a.synthetic:
        ap.error("give --trials all_trials.zip")
    import steps_model as sm                                  # imported here so --help works without PyTorch

    out, models = Path(a.out), Path(a.models)
    out.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- 1. the data ----
    if a.synthetic:
        from synth import make_set
        trials = make_set(a.seed)
        skipped = []
        print("USING SYNTHETIC TRIALS: this checks the machinery, it says nothing about real accuracy.")
    else:
        trials, skipped = load_trials(a.trials)
    for name, why in skipped:
        print("  skipped %s: %s" % (name, why))
    if len(trials) < 8:
        sys.exit("Only %d usable trials: record more with the step recorder (about 40 is a good number)." % len(trials))
    legacy = []
    legacy_dir = a.legacy
    if legacy_dir is None and not a.synthetic:
        for guess in ("../gait-identifier/data", "gait-identifier/data", "data"):
            if Path(guess).is_dir():
                legacy_dir = guess
                break
    if legacy_dir and not a.synthetic:
        legacy = load_legacy_negatives(legacy_dir)
    persons = sorted({t.person for t in trials})
    kinds = sorted({t.kind for t in trials})
    print("%d trials, %d person(s): %s; %.1f minutes, %d steps counted in all" % (len(trials), len(persons), ", ".join(persons), sum(t.seconds for t in trials) / 60, sum(t.steps for t in trials)))
    print("  kinds: " + ", ".join("%s %d" % (k, sum(t.kind == k for t in trials)) for k in kinds))
    print("  plus %d older negative chunks (standing, air-biking) used for training only" % len(legacy))
    true = np.array([t.steps for t in trials], dtype=float)
    k = max(2, min(a.folds, len(trials) // 3))

    # ---- 2. cross-validation: the network and the baseline ----
    print("\nCross-validation (%d folds, %d epochs each): every trial predicted by a model that never saw it" % (k, a.epochs))
    histories, oof_density, restarts = [], {}, []

    def train_nn(train):
        net, hist = sm.fit(train + legacy, epochs=a.epochs, seed=a.seed, log=None)
        histories.append(hist)
        restarts.append(net.restarts)
        return net

    def predict_nn(net, trial):
        d = sm.density_of(net, trial.x)
        oof_density[trial.name] = d
        return float(d.sum())

    nn_pred = ev.cross_validate(trials, train_nn, predict_nn, k=k, seed=a.seed)
    base_pred = ev.cross_validate(trials, lambda tr: Baseline().fit(tr), lambda m, t: m.predict(t), k=k, seed=a.seed)
    m_nn, m_base = ev.metrics(true, nn_pred), ev.metrics(true, base_pred)
    ci_nn, ci_base = ev.bootstrap_ci(true, nn_pred, ev.mae, seed=a.seed), ev.bootstrap_ci(true, base_pred, ev.mae, seed=a.seed)
    ci_nn_pct, ci_base_pct = ev.bootstrap_ci(true, nn_pred, ev.mape, seed=a.seed), ev.bootstrap_ci(true, base_pred, ev.mape, seed=a.seed)
    final_base = Baseline().fit(trials)
    if sum(restarts):
        print("  (%d of %d cross-validation networks got stuck counting nothing and were restarted with another seed)" % (sum(r > 0 for r in restarts), len(restarts)))
    print("  network   average error %.2f steps (95%% interval %.2f to %.2f), %s%% of the count" % (m_nn["mae"], ci_nn[0], ci_nn[1], fmt(m_nn["mape"])))
    print("  baseline  average error %.2f steps (95%% interval %.2f to %.2f), %s%% of the count   [%s]" % (m_base["mae"], ci_base[0], ci_base[1], fmt(m_base["mape"]), final_base.describe()))

    # ---- 3. a new person ----
    person_rows = []
    if len(persons) >= 2:
        for p in persons:
            test = [t for t in trials if t.person == p]
            train = [t for t in trials if t.person != p]
            if len(test) >= 5 and len(train) >= 8:
                net, _ = sm.fit(train + legacy, epochs=a.epochs, seed=a.seed)
                pr = np.array([sm.count_of(net, t.x) for t in test])
                bp = np.array([Baseline().fit(train).predict(t) for t in test])
                person_rows.append({"person": p, "n": len(test), "network": ev.metrics([t.steps for t in test], pr), "baseline": ev.metrics([t.steps for t in test], bp)})
                print("  trained without %s, tested on %s (%d trials): network %.2f steps, baseline %.2f steps" % (p, p, len(test), person_rows[-1]["network"]["mae"], person_rows[-1]["baseline"]["mae"]))

    # ---- 4. distance ----
    dist = {}
    di = [i for i, t in enumerate(trials) if np.isfinite(t.distance_m) and t.steps > 0]
    if len(di) >= 3:
        d_true = np.array([trials[i].distance_m for i in di])
        d_steps = np.array([trials[i].steps for i in di], dtype=float)
        d_act = np.array([active_seconds(oof_density[trials[i].name]) for i in di])
        d_pred_steps = nn_pred[di]
        err_true = leave_one_out(d_steps, d_act, d_true)                     # error from the step LENGTH alone (true step counts)
        e2e = []
        for j in range(len(di)):                                            # end to end: the network's count x the fitted step length
            keep = np.arange(len(di)) != j
            m = DistanceModel().fit(d_steps[keep], d_act[keep], d_true[keep]) if keep.sum() >= 2 else DistanceModel().fit(d_steps, d_act, d_true)
            e2e.append(100.0 * (m.predict(d_pred_steps[j], d_act[j]) - d_true[j]) / d_true[j])
        e2e = np.array(e2e)
        pred_m = d_true * (1 + e2e / 100.0)
        dist = {"n": len(di), "step_length_only_mape": float(np.mean(np.abs(err_true))), "end_to_end_mape": float(np.mean(np.abs(e2e))), "end_to_end_bias_pct": float(np.mean(e2e)),
                "end_to_end_sd_pct": float(np.std(e2e, ddof=1)) if len(e2e) > 1 else 0.0, "end_to_end_worst_pct": float(np.max(np.abs(e2e))), "true_m": d_true.tolist(), "pred_m": pred_m.tolist()}
        per_person = {}
        for p in persons:                                                   # step length differs between people: calibrate each person on their own walks
            idx = [j for j, i in enumerate(di) if trials[i].person == p]
            if len(idx) >= 6:
                st, ac, dt, ps = d_steps[idx], d_act[idx], d_true[idx], d_pred_steps[idx]
                ee = []
                for j in range(len(idx)):
                    keep = np.arange(len(idx)) != j
                    ee.append(100.0 * (DistanceModel().fit(st[keep], ac[keep], dt[keep]).predict(ps[j], ac[j]) - dt[j]) / dt[j])
                ee = np.array(ee)
                per_person[p] = {"n": len(idx), "step_length_only_mape": float(np.mean(np.abs(leave_one_out(st, ac, dt)))), "end_to_end_mape": float(np.mean(np.abs(ee))),
                                 "end_to_end_bias_pct": float(np.mean(ee)), "end_to_end_sd_pct": float(np.std(ee, ddof=1)), "end_to_end_worst_pct": float(np.max(np.abs(ee))),
                                 "model": DistanceModel().fit(st, ac, dt).kind}
        dist["per_person"] = per_person
        print("\nDistance (%d measured walks, each predicted by a model that never saw it): about %.1f%% off using your true step counts, %.1f%% off end to end (one model for everyone)" % (len(di), dist["step_length_only_mape"], dist["end_to_end_mape"]))
        for p, v in per_person.items():
            print("  calibrated on %s's own walks only (%d walks): %.1f%% off using true step counts, %.1f%% off end to end   [%s]" % (p, v["n"], v["step_length_only_mape"], v["end_to_end_mape"], v["model"]))
    else:
        print("\nDistance: fewer than 3 trials with a measured distance, so no distance model (walk the 10 m and 20 m courses and press Edit to enter the distance).")

    # ---- 5. the final model ----
    print("\nTraining the final network on all %d trials (+%d negative chunks)..." % (len(trials), len(legacy)))
    net, hist = sm.fit(trials + legacy, epochs=a.epochs, seed=a.seed, log=print)
    meta = {"trained": datetime.datetime.now().isoformat(timespec="seconds"), "trials": len(trials), "legacy_chunks": len(legacy), "epochs": a.epochs, "seed": a.seed, "synthetic": bool(a.synthetic),
            "cv_mae_steps": m_nn["mae"], "cv_mape_pct": m_nn["mape"], "cv_folds": k, "persons": persons}
    np_net = net.to_numpy(meta)
    np_net.save(models / "step_net.json")
    import torch
    torch.save(net.state_dict(), models / "step_net.pt")
    dm = None
    if len(di) >= 3:
        d_act_final = np.array([active_seconds(sm.density_of(net, trials[i].x)) for i in di])
        dm = DistanceModel().fit(d_steps, d_act_final, d_true)
        print("  distance model: " + dm.kind)
    # the NumPy copy must give the same answer as PyTorch, or the board would count differently from the laptop
    worst = max(abs(np_net.count(__import__("steps_data").features(t.x)) - sm.count_of(net, t.x)) for t in trials[:5])
    print("  NumPy copy vs PyTorch on 5 trials: largest difference %.2g steps" % worst)

    # ---- 6. figures and the summary ----
    tk = [t.kind for t in trials]
    plots.scatter(true, nn_pred, tk, out / "scatter_network.png", "Network, each trial predicted by a model that never saw it")
    plots.scatter(true, base_pred, tk, out / "scatter_baseline.png", "Classical baseline, same test", ylabel="steps the classical counter counted")
    plots.bland_altman(true, nn_pred, out / "errors_network.png", "Network errors")
    plots.loss_curves(histories, out / "training_loss.png")
    plots.compare([("baseline", m_base["mae"], ci_base), ("network", m_nn["mae"], ci_nn)], out / "network_vs_baseline.png")
    order = np.argsort(-np.abs(nn_pred - true))
    plots.example(trials[int(order[len(order) // 2])], oof_density[trials[int(order[len(order) // 2])].name], out / "example_typical.png")
    plots.example(trials[int(order[0])], oof_density[trials[int(order[0])].name], out / "example_worst.png")
    if dist:
        plots.distance_plot(dist["true_m"], dist["pred_m"], out / "distance.png", "Distance: each walk predicted without itself")
    kind_rows = ev.by_kind(trials, nn_pred)
    walk = true > 0
    walking_only = {"n": int(walk.sum()), "network": ev.metrics(true[walk], nn_pred[walk]), "baseline": ev.metrics(true[walk], base_pred[walk])} if walk.any() else {}
    results = {"n_trials": len(trials), "persons": persons, "network": m_nn, "network_mae_ci95": ci_nn, "network_mape_ci95": ci_nn_pct, "baseline": m_base, "baseline_mae_ci95": ci_base,
               "baseline_mape_ci95": ci_base_pct, "baseline_description": final_base.describe(), "by_kind": kind_rows, "walking_only": walking_only, "new_person": person_rows, "distance": dist,
               "oof": [{"name": t.name, "person": t.person, "kind": t.kind, "true": t.steps, "network": float(p), "baseline": float(b)} for t, p, b in zip(trials, nn_pred, base_pred)],
               "numpy_vs_torch_max_diff": float(worst), "synthetic": bool(a.synthetic), "seconds": round(time.time() - t_start),
               "cv_restarts": restarts, "final_restarts": int(net.restarts), "epochs": a.epochs}
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    export.write_assets(models, np_net, trials, results)                     # models/distance.json (everyone and each person) and models/evaluation.json, for the live app
    write_summary(out / "summary.md", results, trials, np_net, dm, a)
    print("\nDone in %d s. Models in %s, figures and summary.md in %s." % (time.time() - t_start, models, out))
    return results


def write_summary(path, r, trials, net, dm, a):
    n, b = r["network"], r["baseline"]
    L = []
    L.append("# Step counter: how good is it?\n")
    if r["synthetic"]:
        L.append("**These numbers come from made-up trials. They check that the code works; they say nothing about real accuracy.**\n")
    L.append("Every number below is out-of-fold: each trial was predicted by a network that never saw it. The truth is the count you typed, and that count can itself be off by a step or two.\n")
    L.append("## Data\n")
    L.append("%d trials, %d person(s), %.1f minutes, %d steps counted.\n" % (r["n_trials"], len(r["persons"]), sum(t.seconds for t in trials) / 60, sum(t.steps for t in trials)))
    L.append("| Kind | Trials | Steps | Network average error (steps) |\n|---|---|---|---|")
    for k, v in r["by_kind"].items():
        L.append("| %s | %d | %d | %.2f |" % (k, v["n"], sum(t.steps for t in trials if t.kind == k), v["mae"]))
    L.append("\n## Network against the classical baseline\n")
    L.append("| | Network | Baseline (%s) |\n|---|---|---|" % r["baseline_description"])
    L.append("| Average error, steps (95%% interval) | **%.2f** (%.2f to %.2f) | %.2f (%.2f to %.2f) |" % (n["mae"], *r["network_mae_ci95"], b["mae"], *r["baseline_mae_ci95"]))
    L.append("| Average error, %% of the count (95%% interval) | **%s%%** (%s to %s) | %s%% (%s to %s) |" % (fmt(n["mape"]), fmt(r["network_mape_ci95"][0]), fmt(r["network_mape_ci95"][1]), fmt(b["mape"]), fmt(r["baseline_mape_ci95"][0]), fmt(r["baseline_mape_ci95"][1])))
    L.append("| Average signed error (bias), steps | %+.2f | %+.2f |" % (n["bias"], b["bias"]))
    L.append("| Spread of the error (SD), steps | %.2f | %.2f |" % (n["sd"], b["sd"]))
    w = r.get("walking_only") or {}
    if w:
        L.append("| Average error on the %d trials WITH steps only, steps | %.2f | %.2f |" % (w["n"], w["network"]["mae"], w["baseline"]["mae"]))
    L.append("| Trials within 1 step | %.0f%% | %.0f%% |" % (100 * n["within1"], 100 * b["within1"]))
    L.append("| Trials within 2 steps | %.0f%% | %.0f%% |" % (100 * n["within2"], 100 * b["within2"]))
    L.append("| Worst trial, steps off | %.1f | %.1f |" % (n["worst"], b["worst"]))
    L.append("| Steps counted in a trial with NO steps (average, worst) | %s, %s | %s, %s |" % (fmt(n["false_steps_mean"]), fmt(n["false_steps_max"]), fmt(b["false_steps_mean"]), fmt(b["false_steps_max"])))
    if r["new_person"]:
        L.append("\n## A new person\n")
        L.append("Trained on everyone else, tested on this person.\n")
        L.append("| Tested on | Trials | Network average error (steps) | Baseline average error (steps) |\n|---|---|---|---|")
        for p in r["new_person"]:
            L.append("| %s | %d | %.2f | %.2f |" % (p["person"], p["n"], p["network"]["mae"], p["baseline"]["mae"]))
    d = r["distance"]
    L.append("\n## Distance walked\n")
    if d:
        L.append("%d walks of a measured distance, each predicted by a model that never saw it. Distance is the number of steps times the length of a step (%s).\n" % (d["n"], dm.kind if dm else "n/a"))
        L.append("| | Average error |\n|---|---|")
        L.append("| Using your true step counts (the step-length model alone) | %.1f%% |" % d["step_length_only_mape"])
        L.append("| End to end (the network's count x step length) | **%.1f%%** (bias %+.1f%%, SD %.1f%%, worst %.1f%%) |" % (d["end_to_end_mape"], d["end_to_end_bias_pct"], d["end_to_end_sd_pct"], d["end_to_end_worst_pct"]))
        for p, v in d.get("per_person", {}).items():
            L.append("| End to end, step length calibrated on %s's own walks only (%d walks; %s) | **%.1f%%** (bias %+.1f%%, SD %.1f%%, worst %.1f%%) |" % (p, v["n"], v["model"], v["end_to_end_mape"], v["end_to_end_bias_pct"], v["end_to_end_sd_pct"], v["end_to_end_worst_pct"]))
        if d.get("per_person"):
            L.append("\nPeople have different step lengths, so one model for everyone does worse than a model calibrated on the wearer's own measured walks.")
    else:
        L.append("Fewer than 3 trials had a measured distance, so no distance model was fitted.\n")
    L.append("\n## Limits to say out loud\n")
    L.append("- The truth is your own count, good to about a step or two, so errors smaller than that are not meaningful.")
    cv = r["cv_restarts"]
    L.append("- Training can get stuck counting nothing. %d of %d cross-validation networks needed a restart with another seed (a stuck run is spotted from the training trials alone: the network counts almost none of their steps. The test trials are never used for that.)" % (sum(1 for x in cv if x > 0), len(cv)))
    L.append("- The training fix (starting the step density very low, so that training does not collapse to counting nothing) was chosen by looking at random splits of these same trials, so the numbers above are slightly optimistic. The first run, before the fix, is described in README.md.")
    L.append("- About %d trials and %d person(s): the intervals above show how much the numbers would move with different trials, and they are wide when there are few." % (r["n_trials"], len(r["persons"])))
    L.append("- The sensor must sit the same way on the same ankle, on the same kind of walking you recorded (corridor, normal shoes). Stairs, slopes, carrying things and other shoes were not tested.")
    L.append("- Step length changes with speed, turns, and the person; the distance error grows with distance and was measured on short straight courses.")
    L.append("\n## Files\n")
    L.append("`scatter_network.png`, `errors_network.png`, `network_vs_baseline.png`, `example_typical.png`, `example_worst.png`, `training_loss.png`, `distance.png`, `results.json` (every trial's prediction).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
