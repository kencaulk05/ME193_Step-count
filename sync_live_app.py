#!/usr/bin/env python3
"""Copy what the live app needs into it, so the app folder is self-contained (App Lab runs an app from its own folder).

    python sync_live_app.py                      after training: puts the network and its step-length models into apps/gait-steps-live
    python sync_live_app.py --check              only checks that the app's copies are up to date (exit 1 if not)
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ["steps_numpy.py", "steps_live.py", "steps_data.py", "steps_distance.py"]       # NumPy only: no PyTorch on the board
MODEL_FILES = ["step_net.json", "distance.json", "evaluation.json"]


def plan(app, models):
    app, models = Path(app), Path(models)
    out = [(HERE / m, app / "python" / m) for m in MODULES]
    out += [(models / f, app / "python" / "model" / f) for f in MODEL_FILES if (models / f).exists()]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default=str(HERE / "apps" / "gait-steps-live"))
    ap.add_argument("--models", default=str(HERE / "models"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if not (Path(a.models) / "step_net.json").exists():
        sys.exit("no trained network in %s: run train_steps.py first" % a.models)
    stale = []
    for src, dst in plan(a.app, a.models):
        same = dst.exists() and dst.read_bytes() == src.read_bytes()
        if not same:
            stale.append(dst)
            if not a.check:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    if a.check:
        print("up to date" if not stale else "OUT OF DATE: " + ", ".join(str(p.name) for p in stale))
        return 1 if stale else 0
    print("copied %d file(s) into %s" % (len(stale), a.app) if stale else "already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
