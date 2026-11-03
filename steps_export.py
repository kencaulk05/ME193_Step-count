"""What the live app needs besides the network: step-length models (one for everyone and one per person) and the headline accuracy numbers.

    distance.json     step length = a + b x cadence, fitted on the measured walks (all of them, and each person's own)
    evaluation.json   the numbers the live page shows as its "how good is this?" card, copied from results.json
"""
import json
from pathlib import Path

import numpy as np

from steps_data import features
from steps_distance import DistanceModel, active_seconds

MIN_WALKS_PER_PERSON = 6


def _walks(np_net, trials):
    """Trials with a steps count and a measured distance, with the time the person was really stepping (from the network's own density)."""
    rows = []
    for t in trials:
        if t.steps > 0 and np.isfinite(t.distance_m):
            rows.append((t.person, float(t.steps), active_seconds(np_net.density(features(t.x))), float(t.distance_m)))
    return rows


def _fit(rows):
    st, ac, dist = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    m = DistanceModel().fit(st, ac, dist)
    return {"a": m.a, "b": m.b, "description": m.kind, "n_trials": len(rows)}


def distance_models(np_net, trials):
    rows = _walks(np_net, trials)
    if len(rows) < 3:
        return None
    out = _fit(rows)
    out["people"] = {p: _fit([r for r in rows if r[0] == p]) for p in sorted({r[0] for r in rows}) if sum(1 for r in rows if r[0] == p) >= MIN_WALKS_PER_PERSON}
    return out


def evaluation_summary(results):
    """The numbers the live page shows. Everything comes from results.json, so it cannot drift from the evaluation."""
    oof = results["oof"]
    t = np.array([r["true"] for r in oof], dtype=float)
    p = np.array([r["network"] for r in oof], dtype=float)
    n, b = results["network"], results["baseline"]
    d = results.get("distance") or {}
    return {"n_trials": results["n_trials"], "persons": results["persons"], "synthetic": bool(results.get("synthetic")),
            "network_mae": n["mae"], "network_mae_ci95": results["network_mae_ci95"], "network_mae_rounded": float(np.mean(np.abs(np.round(p) - t))),
            "within1": n["within1"], "within1_rounded": float(np.mean(np.abs(np.round(p) - t) <= 1)), "worst": n["worst"],
            "false_steps_network": n["false_steps_mean"], "baseline_mae": b["mae"], "false_steps_baseline": b["false_steps_mean"],
            "walking_only_network_mae": (results.get("walking_only") or {}).get("network", {}).get("mae"),
            "walking_only_baseline_mae": (results.get("walking_only") or {}).get("baseline", {}).get("mae"),
            "distance_end_to_end_mape": d.get("end_to_end_mape"), "distance_per_person_mape": {k: v["end_to_end_mape"] for k, v in (d.get("per_person") or {}).items()}}


def write_assets(models_dir, np_net, trials, results=None):
    models_dir = Path(models_dir)
    dm = distance_models(np_net, trials)
    if dm:
        (models_dir / "distance.json").write_text(json.dumps(dm), encoding="utf-8")
    if results is not None:
        (models_dir / "evaluation.json").write_text(json.dumps(evaluation_summary(results)), encoding="utf-8")
    return dm
