#!/usr/bin/env python3
"""Rewrite models/distance.json and models/evaluation.json from an existing trained network and its results, without retraining.

    python export_live_assets.py --trials all_trials.zip
"""
import argparse
import json
from pathlib import Path

from steps_data import load_trials
from steps_export import write_assets
from steps_numpy import NumpyStepNet

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--trials", required=True)
ap.add_argument("--models", default="models")
ap.add_argument("--results", default="results/results.json")
a = ap.parse_args()
net = NumpyStepNet.load(Path(a.models) / "step_net.json")
trials, _ = load_trials(a.trials)
results = json.loads(Path(a.results).read_text(encoding="utf-8"))
dm = write_assets(a.models, net, trials, results)
print("wrote %s/distance.json (%s%s) and %s/evaluation.json" % (a.models, dm["description"] if dm else "no distance model", "; people: " + ", ".join("%s %s" % (k, v["description"]) for k, v in dm["people"].items()) if dm else "", a.models))
