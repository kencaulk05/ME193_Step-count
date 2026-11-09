"""Tests for the step counter (steps_*.py, train_steps.py). No board needed.   python tests/test_steps.py
The synthetic trials in these tests have a known right answer by construction: they check the MACHINERY, not real accuracy."""
import csv, io, json, os, re, subprocess, sys, tempfile, zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
problems = 0


def expect(label, cond, detail=""):
    global problems
    print(f"  {'ok  ' if cond else 'FAIL'} {label}" + (f"   [{detail}]" if detail else ""))
    problems += (not cond)


import steps_data as sd
import steps_eval as ev
from steps_baseline import Baseline, count_steps
from steps_distance import DistanceModel, active_seconds, leave_one_out
from steps_numpy import NumpyStepNet, conv_same
from synth import make_set, make_trial

try:
    import torch
    import steps_model as sm
    HAVE_TORCH = True
except Exception as e:                                   # the logic above still gets tested without PyTorch
    HAVE_TORCH = False
    print("(PyTorch is not installed here: its tests are skipped: %s)" % str(e)[:60])

LEGACY = None
for guess in (ROOT.parent / "gait-identifier" / "data", ROOT / "data", Path("/tmp/gi/gait-identifier/data")):
    if guess.is_dir() and list(guess.glob("*__still__*.csv")):
        LEGACY = guess
        break


def recorder_zip(path, rows, files):
    """Write what the step recorder's 'Prepare zip of everything' button writes."""
    fields = ["file", "subject", "kind", "true_steps", "distance_m", "note", "start", "seconds", "samples", "dropped", "lost_pct"]
    with zipfile.ZipFile(path, "w") as z:
        buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=fields); w.writeheader(); w.writerows(rows)
        z.writestr("manifest.csv", buf.getvalue())
        for name, arr in files.items():
            lines = ["n,ax,ay,az,gx,gy,gz"] + ["%d,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f" % ((i,) + tuple(r)) for i, r in enumerate(arr)]
            z.writestr(name, "\n".join(lines) + "\n")


def row(file, subject="ann", kind="walk_normal", steps="20", dist="", lost=0.0):
    return {"file": file, "subject": subject, "kind": kind, "true_steps": steps, "distance_m": dist, "note": "", "start": "2026-11-02 14:00:00", "seconds": 5, "samples": 500, "dropped": 0, "lost_pct": lost}


rng0 = np.random.default_rng(1)
tmp = Path(tempfile.mkdtemp())

print("1. loading what the step recorder saved")
sig = rng0.normal(0, 1, (500, 6)).astype(np.float32)
recorder_zip(tmp / "t.zip", [row("a.csv", steps="20", dist="10"), row("b.csv", kind="foot_tap", steps="0"), row("c.csv", steps=""), row("d.csv", lost=5.0), row("gone.csv")],
             {"a.csv": sig, "b.csv": sig, "c.csv": sig, "d.csv": sig})
trials, skipped = sd.load_trials(tmp / "t.zip")
expect("two usable trials are loaded; the other three are skipped, each with its reason", [t.name for t in trials] == ["A__a.csv", "A__b.csv"] and len(skipped) == 3, str(skipped))
reasons = dict(skipped)
expect("the reasons are in plain words: Edit for a missing count, lost readings, a missing file", "Edit" in reasons["A__c.csv"] and "lost" in reasons["A__d.csv"] and "missing" in reasons["A__gone.csv"], str(reasons))
t = trials[0]
expect("a trial carries person, kind, steps, distance and a (T, 6) float32 signal", (t.person, t.kind, t.steps, t.distance_m) == ("A", "walk_normal", 20, 10.0) and t.x.shape == (500, 6) and t.x.dtype == np.float32 and abs(t.seconds - 5.0) < 1e-9)
expect("no distance means NaN, and a 0-step trial is kept", np.isnan(trials[1].distance_m) and trials[1].steps == 0)
(tmp / "folder").mkdir()
with zipfile.ZipFile(tmp / "t.zip") as z: z.extractall(tmp / "folder")
t2, _ = sd.load_trials(tmp / "folder")
expect("a plain folder works the same as the zip", [x.name for x in t2] == ["A__a.csv", "A__b.csv"] and np.allclose(t2[0].x, trials[0].x))
with zipfile.ZipFile(tmp / "nom.zip", "w") as z: z.writestr("a.csv", "n,ax,ay,az,gx,gy,gz\n0,0,0,0,0,0,0\n")
try:
    sd.load_trials(tmp / "nom.zip"); expect("a zip with no manifest is refused", False)
except ValueError as e:
    expect("a zip with no manifest is refused, and says which zip to use", "manifest.csv" in str(e) and "Prepare zip" in str(e), str(e)[:80])
with zipfile.ZipFile(tmp / "bad.zip", "w") as z:
    z.writestr("manifest.csv", "file,subject,kind,true_steps,distance_m,note,start,seconds,samples,dropped,lost_pct\nx.csv,ann,walk_normal,20,,,2026-11-02 14:00:00,5,500,0,0\n")
    z.writestr("x.csv", "a,b,c\n1,2,3\n")
try:
    sd.load_trials(tmp / "bad.zip"); expect("a CSV with the wrong header is refused", False)
except ValueError as e:
    expect("a CSV with the wrong header is refused", "not a recorder file" in str(e))
ul, _ = sd.load_trials(tmp / "t.zip", include_unlabelled=True)
expect("trials still waiting for their count can be loaded on request (steps -1)", [x.steps for x in ul if x.name == "A__c.csv"] == [-1])

print("1b. nobody is named: people become letters")
an = sd.Anonymizer()
expect("people become letters in the order they first appear, and the same person (any capitals, any spaces) always gets the same letter", [an.label(n) for n in ("Zoe", "adam", " ZOE ", "Adam", "kim")] == ["A", "B", "A", "B", "C"])
expect("after Z comes AA, AB, ...", sd._letters(25) == "Z" and sd._letters(26) == "AA" and sd._letters(27) == "AB" and sd._letters(51) == "AZ" and sd._letters(52) == "BA")
a2 = sd.Anonymizer({"Zoe": "B"})
expect("an alias file chooses letters, and the others get the next unused ones", [a2.label("zoe"), a2.label("adam"), a2.label("kim")] == ["B", "A", "C"])
expect("keep=True switches it off (for private use)", sd.Anonymizer(keep=True).label("Zoe") == "Zoe" and sd.Anonymizer(keep=True).file_name("Zoe__walk.csv", "Zoe") == "Zoe__walk.csv")
fn = lambda f, p: sd.Anonymizer().file_name(f, p)
expect("in a recorder file name the name is swapped for the letter, whatever its capitals, and a name with a space (saved with an underscore) too", fn("Zoe__walk_normal__20steps__20261102-140007.csv", "zoe") == "A__walk_normal__20steps__20261102-140007.csv" and fn("sam_j__live__2026.csv", "Sam J") == "A__live__2026.csv")
expect("a file name that does not start with the name still never keeps it: the letter goes in front of the whole name only when the name is not in it", fn("odd.csv", "zoe") == "A__odd.csv")
with zipfile.ZipFile(tmp / "named.zip", "w") as z:
    fields = ["file", "subject", "kind", "true_steps", "distance_m", "note", "start", "seconds", "samples", "dropped", "lost_pct"]
    z.writestr("manifest.csv", ",".join(fields) + "\nZoe__walk_normal__20steps__20261102-140007.csv,Zoe,walk_normal,20,,,2026-11-02 14:00:07,5,500,0,0\nadam__walk_fast__20steps__20261102-140100.csv,adam,walk_fast,20,,,2026-11-02 14:01:00,5,500,0,0\nzoe__stand__0steps__20261102-140200.csv,zoe,stand,0,,,2026-11-02 14:02:00,5,500,0,0\nadam__stand__unknownsteps__20261102-140300.csv,adam,stand,,,,2026-11-02 14:03:00,5,500,0,0\n")
    for nm in ("Zoe__walk_normal__20steps__20261102-140007.csv", "adam__walk_fast__20steps__20261102-140100.csv", "zoe__stand__0steps__20261102-140200.csv", "adam__stand__unknownsteps__20261102-140300.csv"):
        z.writestr(nm, "n,ax,ay,az,gx,gy,gz\n" + "\n".join("%d,0,0,-1,0,0,0" % i for i in range(300)) + "\n")
nt, ns = sd.load_trials(tmp / "named.zip")
expect("loading a zip with real names: the people are A and B, in order of first appearance, and the same person is the same letter in every trial", [t.person for t in nt] == ["A", "B", "A"], str([t.person for t in nt]))
expect("no trial name and no skipped-trial message keeps a name", all(re.fullmatch(r"[A-Z]{1,2}__.*", t.name) for t in nt) and all(re.fullmatch(r"[A-Z]{1,2}__.*", n) for n, _ in ns) and not any(("zoe" in t.name.lower() or "adam" in t.name.lower()) for t in nt) and not any(("zoe" in n.lower() or "adam" in n.lower()) for n, _ in ns), str([t.name for t in nt] + [n for n, _ in ns]))
shared = sd.Anonymizer()
nt2, _ = sd.load_trials(tmp / "named.zip", anonymizer=shared)
(tmp / "legacy_named").mkdir(exist_ok=True)
for who in ("Zoe", "Adam"):
    (tmp / "legacy_named" / ("%s__still__20261027-031222.csv" % who)).write_text("n,ax,ay,az,gx,gy,gz\n" + "\n".join("%d,0,0,-1,0,0,0" % i for i in range(3200)) + "\n")
lg = sd.load_legacy_negatives(tmp / "legacy_named", anonymizer=shared)
expect("the older recordings, loaded with the same anonymizer, use the same letters for the same people (Zoe is A, Adam is B) and carry no name", {l.person for l in lg} == {"A", "B"} and all(not ("zoe" in l.name.lower() or "adam" in l.name.lower()) for l in lg), str([l.name for l in lg][:2]))
nk, _ = sd.load_trials(tmp / "named.zip", anonymizer=sd.Anonymizer(keep=True))
expect("with keep=True the names stay (private use)", [t.person for t in nk] == ["Zoe", "adam", "zoe"])

print("2. the older recordings as free negative examples")
if LEGACY:
    neg = sd.load_legacy_negatives(LEGACY)
    kinds = sorted({n.kind for n in neg})
    expect("standing and air-biking are used, walking and running are not", kinds == ["bike", "stand"], str(kinds))
    expect("every one has 0 steps and is marked as older data", all(n.steps == 0 and n.source == "legacy" for n in neg) and len(neg) >= 24, str(len(neg)))
    expect("each is a chunk of at least 15 s, and the first and last second of every recording are dropped", all(len(n.x) >= 1500 for n in neg) and all(len(n.x) <= 3000 for n in neg))
else:
    print("  (the older recordings were not found, so this group is skipped)")

print("3. the numbers the network reads")
g = np.tile(np.array([[-0.95, 0, 0, 0, 0, 0]], dtype=np.float32), (200, 1))
f = sd.features(g)
expect("8 numbers per reading, finite", f.shape == (200, 8) and np.isfinite(f).all())
expect("gravity along the shin: ax is -0.95 g, shown in units of 4 g; total acceleration minus 1 g is tiny; no rotation", abs(f[0, 0] + 0.2375) < 1e-4 and abs(f[0, 6] - (0.95 - 1.0) / 2) < 1e-4 and abs(f[0, 7]) < 1e-6)
r = np.zeros((10, 6), dtype=np.float32); r[:, 3] = 300.0
expect("300 degrees/s on gx shows as 0.6 (units of 500), and as 0.6 total rotation", abs(sd.features(r)[0, 3] - 0.6) < 1e-6 and abs(sd.features(r)[0, 7] - 0.6) < 1e-6)

print("4. the NumPy copy of the network")
w = rng0.normal(0, 1, (3, 2, 5)); b = rng0.normal(0, 1, 3); x = rng0.normal(0, 1, (40, 2))
def naive(x, w, b, d):
    T, K = len(x), w.shape[2]; out = np.zeros((T, w.shape[0])); pad = d * (K - 1) // 2
    for t in range(T):
        for o in range(w.shape[0]):
            s = b[o]
            for i in range(w.shape[1]):
                for k in range(K):
                    j = t + k * d - pad
                    if 0 <= j < T: s += w[o, i, k] * x[j, i]
            out[t, o] = s
    return out
expect("the convolution equals a slow loop-by-loop version, for dilations 1, 3 and 8", all(np.allclose(conv_same(x, w, b, d), naive(x, w, b, d)) for d in (1, 3, 8)))
layers = [{"w": rng0.normal(0, .3, (4, 8, 5)), "b": rng0.normal(0, .1, 4), "dilation": 1}, {"w": rng0.normal(0, .3, (4, 4, 5)), "b": rng0.normal(0, .1, 4), "dilation": 2}]
net_np = NumpyStepNet(layers, rng0.normal(0, .3, (1, 4, 1)), rng0.normal(0, .1, 1), {"note": "x"})
feats = rng0.normal(0, 1, (120, 8)); dens = net_np.density(feats)
expect("the density is never negative, one number per reading", dens.shape == (120,) and (dens >= 0).all())
expect("the count is the sum of the density", abs(net_np.count(feats) - dens.sum()) < 1e-12)
expect("the receptive field is 1 + 4 x (1 + 2) = 13 readings", net_np.receptive_field == 13)
net_np.save(tmp / "n.json"); back = NumpyStepNet.load(tmp / "n.json")
expect("saved and loaded, it gives exactly the same density, and keeps its notes", np.allclose(back.density(feats), dens) and back.meta == {"note": "x"} and back.n_parameters() == net_np.n_parameters())

print("5. the classical baseline (absolute thresholds)")
if LEGACY:
    negs = sd.load_legacy_negatives(LEGACY)
    stills = [n for n in negs if n.kind == "stand"]
    bike = next(n for n in negs if n.kind == "bike")
    walk = sd._read_csv_text(next(iter(sorted(LEGACY.glob("*__walk__*.csv")))).read_text(encoding="utf-8"), "walk")[200:-200]
    expect("on all %d REAL motionless chunks, absolute thresholds count no steps (a relative one counted over 100 per recording)" % len(stills), all(count_steps(n.x, "gyro", 50, 0.6) == 0 and count_steps(n.x, "acc", 0.12, 0.3) == 0 for n in stills))
    expect("but a classical counter DOES count REAL air-biking as steps (%d), which is what the network must learn not to do" % count_steps(bike.x, "gyro", 50, 0.6), count_steps(bike.x, "gyro", 50, 0.6) > 10)
    expect("on REAL walking it counts plenty", count_steps(walk, "gyro", 50, 0.6) > 20, "%d" % count_steps(walk, "gyro", 50, 0.6))
syn = make_set(5, n_walk=24, n_neg=9)
base = Baseline().fit(syn)
errs = [abs(base.predict(t) - t.steps) for t in syn]
expect("fitted on synthetic trials it picks a family, a threshold and a gap, and describes them", base.family in ("gyro", "acc") and base.threshold > 0 and "peaks above" in base.describe(), base.describe())
expect("and counts zero steps on every synthetic standing trial", all(base.predict(t) == 0 for t in syn if t.kind == "stand"))
expect("its training error is what fit() reports", abs(np.mean(errs) - base.train_mae) < 1e-9)

print("6. distance")
st = np.array([20, 26, 31, 15, 40, 22, 35, 28.0]); act = st / np.array([1.4, 1.7, 2.0, 1.2, 2.3, 1.5, 2.1, 1.8])
true_len = 0.30 + 0.17 * (st / act); dist = st * true_len
m = DistanceModel().fit(st, act, dist)
expect("it recovers step length = 0.30 + 0.17 x cadence from exact data", abs(m.a - 0.30) < 1e-6 and abs(m.b - 0.17) < 1e-6 and "cadence" in m.kind, m.kind)
expect("and predicts a new walk", abs(m.predict(30, 15) - 30 * (0.30 + 0.17 * 2.0)) < 1e-6)
m2 = DistanceModel().fit(st[:4], act[:4], dist[:4])
expect("with fewer than 6 walks it falls back to one constant step length", m2.b == 0.0 and "constant" in m2.kind)
expect("a silly step length is clamped to 0.3 to 1.2 m", m.step_length(10, 1.0) <= 1.2 and m.step_length(10, 1000.0) >= 0.3)
loo = leave_one_out(st, act, dist)
expect("leave-one-out on exact data: every walk within 0.5%", len(loo) == 8 and np.max(np.abs(loo)) < 0.5, "%.3f%%" % np.max(np.abs(loo)))
d = np.zeros(1000); d[200:700] = 0.04
expect("the active time of a density that is on for 5 s is about 4.5 s (from 5% to 95% of the steps)", 4.2 <= active_seconds(d) <= 4.8, "%.2f" % active_seconds(d))
expect("a density with almost nothing in it falls back to the trial length", active_seconds(np.full(500, 1e-5)) == 5.0)

print("7. the evaluation tools")
tr = make_set(3, n_walk=20, n_neg=10)
folds = ev.make_folds(tr, 5, seed=0)
flat = sorted(i for f in folds for i in f)
expect("every trial is in exactly one fold", flat == list(range(len(tr))))
expect("every fold has a walk and a negative trial (so each is a fair test)", all(any(tr[i].steps > 0 for i in f) and any(tr[i].steps == 0 for i in f) for f in folds))
expect("the folds are the same for the same seed and different for another", ev.make_folds(tr, 5, 0) == folds and ev.make_folds(tr, 5, 1) != folds)
m = ev.metrics([20, 20, 0, 40], [22, 18, 1, 40])
expect("metrics: average error 1.25, bias 0.25, within 1 step 50%, false steps on the 0-step trial 1", abs(m["mae"] - 1.25) < 1e-9 and abs(m["bias"] - 0.25) < 1e-9 and m["within1"] == 0.5 and m["false_steps_mean"] == 1.0 and m["n_zero"] == 1, str(m))
expect("percentage error ignores the 0-step trials: (2/20 + 2/20 + 0)/3 = 6.67%", abs(m["mape"] - 100 * (0.1 + 0.1 + 0.0) / 3) < 1e-9, "%.3f" % m["mape"])
tt, pp = np.arange(10, 60, 5.0), np.arange(10, 60, 5.0) + np.tile([1.0, -1.0], 5)
ci = ev.bootstrap_ci(tt, pp, ev.mae, seed=0)
expect("the bootstrap interval brackets the average error (1.0), and is repeatable", ci[0] <= 1.0 <= ci[1] and ci == ev.bootstrap_ci(tt, pp, ev.mae, seed=0), str(ci))
expect("by kind: one row per kind, with its own error", set(ev.by_kind(tr, [t.steps for t in tr]).keys()) == {t.kind for t in tr})
oof = ev.cross_validate(tr, lambda train: float(np.mean([t.steps for t in train])), lambda model, t: model, k=5)
expect("cross-validation fills in a prediction for every trial, from a model that did not see it", len(oof) == len(tr) and np.isfinite(oof).all())


print("7b. counting while walking (the live counter)")
from steps_live import LiveCounter
rl = np.random.default_rng(2)
def random_net(seed, dil=(1, 2, 4, 8, 16), width=8):
    r = np.random.default_rng(seed); layers, c = [], 8
    for d in dil:
        layers.append({"w": r.normal(0, 0.25, (width, c, 5)), "b": r.normal(0, 0.05, width), "dilation": d}); c = width
    return NumpyStepNet(layers, r.normal(0, 0.3, (1, c, 1)), np.array([-3.5]))
rnet = random_net(7)
xs = make_trial(np.random.default_rng(4), "walk_normal", steps=22).x
offline = rnet.count(sd.features(xs))
expect("the receptive field is 125 readings, so it looks 62 readings (0.62 s) each way", rnet.receptive_field == 125 and LiveCounter(rnet).half == 62)
live = LiveCounter(rnet); i = 0; committed = [0.0]; chunk = np.random.default_rng(9)
while i < len(xs):
    k = int(chunk.integers(1, 81)); live.push(xs[i:i + k]); i += k; committed.append(live.committed)
final = live.finish()
expect("pushed in random chunks of 1 to 80 readings, the final live count EQUALS the offline count of the whole walk (%.6f)" % offline, abs(final - offline) < 1e-9, "%.9f vs %.9f" % (final, offline))
expect("the committed count only ever goes up", all(b >= a - 1e-12 for a, b in zip(committed, committed[1:])))
live2 = LiveCounter(rnet, update_every=25); worst = 0.0
for j in range(0, len(xs) - 24, 25):
    live2.push(xs[j:j + 25]); worst = max(worst, abs(live2.count - rnet.count(sd.features(xs[:j + 25]))))
expect("after every update, the on-screen count equals the offline count of everything heard so far (largest gap %.1e)" % worst, worst < 1e-9)
one = LiveCounter(rnet)
for j in range(len(xs)): one.push(xs[j])
expect("one reading at a time gives the same final count", abs(one.finish() - offline) < 1e-9)
expect("finishing twice changes nothing", abs(one.finish() - offline) < 1e-9)
try:
    one.push(xs[0]); expect("pushing after finish is refused", False)
except RuntimeError:
    expect("pushing after finish is refused, and says to reset", True)
one.reset()
expect("reset starts a new walk from zero", one.count == 0.0 and one.n == 0 and abs(one.push(xs[:200]) - 0) >= 0)
# a network whose answer is known exactly: 2 steps a second whatever the signal (density 0.02 per reading)
zero_layers = [{"w": np.zeros((4, 8 if i == 0 else 4, 5)), "b": np.zeros(4), "dilation": d} for i, d in enumerate((1, 2, 4, 8, 16))]
const = NumpyStepNet(zero_layers, np.zeros((1, 4, 1)), np.array([np.log(np.expm1(0.02))]))
c = LiveCounter(const, update_every=10); xx = np.random.default_rng(1).normal(0, 1, (1000, 6)).astype(np.float32); errs = []
for j in range(0, 1000, 10):
    c.push(xx[j:j + 10]); errs.append(abs(c.count - 0.02 * (j + 10)))
expect("with a network that always says 2 steps a second, the live count is exactly 0.02 x readings after every update (largest gap %.1e)" % max(errs), max(errs) < 1e-9)
lag = LiveCounter(const); lag.push(xx[:24])
expect("it refreshes every 25 readings (0.25 s): after 24 it has not updated yet, after 25 it has", lag.count == 0.0 and abs(lag.push(xx[24:25]) - 0.02 * 25) < 1e-9)
expect("and at the end it is 20.000 steps for 10 seconds", abs(c.finish() - 20.0) < 1e-9)
expect("the pace reads 120 steps a minute", abs(c.steps_per_minute() - 120.0) < 1e-6, "%.3f" % c.steps_per_minute())
expect("and the active time is about 9 s (from 5% to 95% of the steps)", 8.5 <= c.active_seconds() <= 9.5, "%.2f" % c.active_seconds())
short = LiveCounter(const); short.push(xx[:50])
expect("a pace needs at least a second of final density: before that it says 0", short.steps_per_minute() == 0.0)

print("7c. the step-length models and the evaluation card the live app needs")
import steps_export as ex
rn = random_net(3)
walks = []
for i, (person, steps, dist) in enumerate([("ann", 20, 10.0), ("ann", 26, 13.4), ("ann", 15, 7.4), ("ann", 31, 16.5), ("ann", 22, 11.2), ("ann", 28, 14.9), ("bob", 20, 9.0), ("bob", 18, 8.2)]):
    t = make_trial(np.random.default_rng(i), "walk_normal", steps=steps, person=person); t.distance_m = dist; walks.append(t)
dmods = ex.distance_models(rn, walks)
expect("a pooled model, plus one for each person with at least 6 walks (ann yes, bob has only 2)", dmods["n_trials"] == 8 and set(dmods["people"]) == {"ann"} and dmods["people"]["ann"]["n_trials"] == 6, str(dmods["people"].keys()))
expect("each model is a line: a, b and a description that says so", all(k in dmods for k in ("a", "b", "description")) and "cadence" in dmods["people"]["ann"]["description"] or "constant" in dmods["people"]["ann"]["description"])
expect("with fewer than 3 measured walks there is no model at all", ex.distance_models(rn, walks[:2]) is None)
oof = [{"true": 10, "network": 10.4}, {"true": 12, "network": 11.6}, {"true": 0, "network": 0.0}, {"true": 20, "network": 22.6}]
res = {"n_trials": 4, "persons": ["ann"], "network": {"mae": 0.85, "within1": 0.75, "worst": 2.6, "false_steps_mean": 0.0}, "network_mae_ci95": [0.1, 1.5], "baseline": {"mae": 1.2, "false_steps_mean": 0.5}, "oof": oof,
       "walking_only": {"network": {"mae": 1.1}, "baseline": {"mae": 0.9}}, "distance": {"end_to_end_mape": 11.0, "per_person": {"ann": {"end_to_end_mape": 7.0}}}}
card = ex.evaluation_summary(res)
expect("the card copies the headline numbers and adds the error once the count is rounded to a whole step (10.4 -> 10, 11.6 -> 12, 0 -> 0, 22.6 -> 23: errors 0, 0, 0, 3)", abs(card["network_mae_rounded"] - 0.75) < 1e-9 and card["within1_rounded"] == 0.75 and card["network_mae"] == 0.85 and card["distance_per_person_mape"] == {"ann": 7.0}, str(card))
out_dir = tmp / "assets_out"; out_dir.mkdir()
ex.write_assets(out_dir, rn, walks, res)
expect("write_assets writes distance.json and evaluation.json, and both read back as JSON", json.loads((out_dir / "distance.json").read_text())["n_trials"] == 8 and json.loads((out_dir / "evaluation.json").read_text())["n_trials"] == 4)

print("7d. copying the network into the live app")
import sync_live_app as sy
app_dir, mdl = tmp / "app", tmp / "mdl"; mdl.mkdir()
rn.save(mdl / "step_net.json"); (mdl / "distance.json").write_text("{}"); (mdl / "evaluation.json").write_text("{}")
expect("the app needs only NumPy modules and the model files: no PyTorch module is copied", all("torch" not in (ROOT / m).read_text() for m in sy.MODULES) and "steps_model.py" not in sy.MODULES)
expect("--check on an empty app says it is out of date (exit 1), and copies nothing", sy.main(["--app", str(app_dir), "--models", str(mdl), "--check"]) == 1 and not (app_dir / "python").exists())
expect("syncing copies the modules and the model", sy.main(["--app", str(app_dir), "--models", str(mdl)]) == 0 and all((app_dir / "python" / m).exists() for m in sy.MODULES) and (app_dir / "python" / "model" / "step_net.json").exists())
expect("afterwards --check says up to date (exit 0)", sy.main(["--app", str(app_dir), "--models", str(mdl), "--check"]) == 0)
(app_dir / "python" / "steps_live.py").write_text("# edited by hand")
expect("a copy changed by hand is noticed (exit 1) and repaired by the next sync", sy.main(["--app", str(app_dir), "--models", str(mdl), "--check"]) == 1 and sy.main(["--app", str(app_dir), "--models", str(mdl)]) == 0 and (app_dir / "python" / "steps_live.py").read_bytes() == (ROOT / "steps_live.py").read_bytes())
try:
    sy.main(["--app", str(app_dir), "--models", str(tmp / "nothing")]); expect("syncing with no trained network is refused", False)
except SystemExit as e:
    expect("syncing with no trained network is refused, and says to run train_steps.py", "train_steps.py" in str(e))
real_app = ROOT / "apps" / "gait-steps-live"
if real_app.exists() and (ROOT / "models" / "step_net.json").exists():
    expect("the live app in this project carries the CURRENT network and modules (run sync_live_app.py after retraining)", sy.main(["--app", str(real_app), "--models", str(ROOT / "models"), "--check"]) == 0)

print("7e. nothing that ships carries a name")
shipped_ok = True
res_path, mdl_dir = ROOT / "results" / "results.json", ROOT / "models"
if res_path.exists() and not json.loads(res_path.read_text(encoding="utf-8")).get("synthetic"):
    RJ = json.loads(res_path.read_text(encoding="utf-8"))
    expect("in the shipped results.json every person is a letter, and every trial name starts with that letter and two underscores", all(re.fullmatch(r"[A-Z]{1,2}", p) for p in RJ["persons"]) and all(re.fullmatch(r"[A-Z]{1,2}", r["person"]) and re.match(r"[A-Z]{1,2}__", r["name"]) for r in RJ["oof"]), str(RJ["persons"]))
    expect("the people tested as a new person, and in the distance results, are letters too", all(re.fullmatch(r"[A-Z]{1,2}", p["person"]) for p in RJ["new_person"]) and all(re.fullmatch(r"[A-Z]{1,2}", k) for k in (RJ.get("distance") or {}).get("per_person", {})))
    texts = {p.name: p.read_text(encoding="utf-8") for p in list((ROOT / "results").glob("*.md")) + [ROOT / "README.md"]}
    bad = [(n, m) for n, tx in texts.items() for m in re.findall(r"(\w+)__(?:walk_slow|walk_normal|walk_fast|jog|foot_tap|stand|other|bike)", tx) if not re.fullmatch(r"[A-Z]{1,2}", m)]
    expect("no file name in the shipped docs or summaries has a person's name in front of it", not bad, str(bad[:3]))
if (mdl_dir / "distance.json").exists() and (mdl_dir / "evaluation.json").exists():
    DJ, EJ = json.loads((mdl_dir / "distance.json").read_text(encoding="utf-8")), json.loads((mdl_dir / "evaluation.json").read_text(encoding="utf-8"))
    SJ = json.loads((mdl_dir / "step_net.json").read_text(encoding="utf-8"))
    syn = SJ.get("meta", {}).get("synthetic")
    if not syn:
        expect("in the shipped models the per-person step lengths, the accuracy card and the training notes use letters only", all(re.fullmatch(r"[A-Z]{1,2}", k) for k in DJ.get("people", {})) and all(re.fullmatch(r"[A-Z]{1,2}", k) for k in EJ.get("distance_per_person_mape", {})) and all(re.fullmatch(r"[A-Z]{1,2}", p) for p in SJ["meta"].get("persons", [])) and all(re.fullmatch(r"[A-Z]{1,2}", p) for p in EJ.get("persons", [])), str(list(DJ.get("people", {}))))
for ignore in (ROOT / ".gitignore", ROOT / "apps" / "gait-steps-live" / ".gitignore"):
    expect("%s keeps local recordings and the alias file out of git" % ignore.relative_to(ROOT), ignore.exists() and ("assets/data/" in ignore.read_text() or "all_trials" in ignore.read_text()))

print("7f. the illustrations and the README")
IMG = ["system-overview", "data-flow", "signal-anatomy", "architecture", "receptive-field", "count-is-a-sum", "training-loop", "collapse-and-fix", "live-counter-timeline", "results-overview", "architecture-comparison", "distance-model", "led-matrix", "decisions"]
readme = (ROOT / "README.md").read_text(encoding="utf-8")
linked = re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", readme)
expect("every picture the README links to exists, is a real PNG, and is small enough for Notion's 5 MB cap and for git (under 1 MB)", linked and all((ROOT / l).exists() and (ROOT / l).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and 5_000 < (ROOT / l).stat().st_size < 1_000_000 for l in linked), str([l for l in linked if not (ROOT / l).exists()]))
expect("all %d illustrations are in docs/images, and the README uses every one" % len(IMG), all((ROOT / "docs" / "images" / (n + ".png")).exists() for n in IMG) and all(("docs/images/%s.png" % n) in readme for n in IMG))
nums = json.loads((ROOT / "docs" / "architecture_comparison_numbers.json").read_text(encoding="utf-8"))
expect("the comparison numbers say where they came from, start with the shipped network, and each average lies inside its interval", "log" in nums["source"] and "GRU" in nums["source"] and nums["variants"][0]["name"] == "reference" and all(v["lo"] <= v["mean"] <= v["hi"] for v in nums["variants"]) and len(nums["variants"]) == 7)
expect("the README says plainly that the GRU was not completed, that the paired intervals were not computed, and that the numbers are slightly optimistic", "GRU is missing" in readme and "never computed" in readme and "slightly optimistic" in readme)
expect("the table in the README matches the numbers file, variant by variant", all(("%.2f (%.2f to %.2f)" % (v["mean"], v["lo"], v["hi"])) in readme for v in nums["variants"]))
expect("the README has no personal name in it", not re.search(r"(?<![a-z])(" + "ken" "drick" + "|" + "ma" "rie" + ")(?![a-z])", readme, re.I))
import importlib.util as _iu
_sp = _iu.spec_from_file_location("make_images", ROOT / "docs" / "make_images.py"); _mi = _iu.module_from_spec(_sp); _sp.loader.exec_module(_mi)
expect("the LED figure is drawn from the sketch's own font and lights the same number of LEDs the firmware test expects (7: 11, 42: 28, 99: 30, 100 or more: 43)", _mi.led_digits() == {7: 11, 42: 28, 99: 30, 150: 43})

if not HAVE_TORCH:
    print("\nPyTorch tests skipped.")
    print("ALL PASSED (without PyTorch)" if problems == 0 else f"{problems} CHECK(S) FAILED")
    sys.exit(1 if problems else 0)

print("8. the PyTorch network")
net = sm.StepNet()
n_par = sum(p.numel() for p in net.parameters())
expect("about 6000 weights (small enough for a few dozen trials)", 5000 < n_par < 7000, str(n_par))
xx = torch.randn(3, 8, 400)
dd = net(xx)
expect("one density per reading, never negative", tuple(dd.shape) == (3, 400) and bool((dd >= 0).all()))
expect("it starts out guessing a tiny density, about 0.001 steps per reading (0.1 a second), before any training: starting high is what made it collapse", 0.0003 < float(dd.detach().mean()) < 0.004, "%.5f" % float(dd.detach().mean()))
npn = net.to_numpy()
f = np.random.default_rng(0).normal(0, 1, (400, 8)).astype(np.float32)
with torch.no_grad(): pt = net(torch.from_numpy(f.T[None].copy()))[0].numpy()
expect("the NumPy copy gives the same density as PyTorch (largest difference under 1e-5)", np.max(np.abs(npn.density(f) - pt)) < 1e-5, "%.2e" % np.max(np.abs(npn.density(f) - pt)))
expect("receptive field of 125 readings, 1.25 seconds", npn.receptive_field == 125)
expect("the NumPy copy has the same number of weights", npn.n_parameters() == n_par)

print("9. the loss")
L = lambda p, y: float(sm.count_loss(torch.tensor(p), torch.tensor(y)))
expect("a perfect count costs nothing", L([20.0], [20.0]) == 0.0)
expect("25 counted when it was 20 costs ln(26/21) = 0.214", abs(L([25.0], [20.0]) - np.log(26 / 21)) < 1e-6, "%.4f" % L([25.0], [20.0]))
expect("1 step counted on a 0-step trial costs ln 2 = 0.693: more than being 5 steps out on a walk, so false steps are punished", abs(L([1.0], [0.0]) - np.log(2)) < 1e-6 and L([1.0], [0.0]) > L([25.0], [20.0]))
expect("but counting 60 steps on a 0-step trial costs 4.1, not 12: the penalty is bounded, which is what stops the network collapsing to zero", 3.5 < L([60.0], [0.0]) < 4.5, "%.2f" % L([60.0], [0.0]))
expect("the loss is the average over the trials", abs(L([25.0, 20.0], [20.0, 20.0]) - np.log(26 / 21) / 2) < 1e-6)

print("10. the augmentations keep the count valid")
rng = np.random.default_rng(0); x0 = make_trial(rng, "walk_normal", steps=20).x
lens = [len(sm.augment(x0, np.random.default_rng(s))) / len(x0) for s in range(40)]
expect("the same steps faster or slower: the length changes by 0.8 to 1.25 times", 0.79 <= min(lens) and max(lens) <= 1.26 and max(lens) - min(lens) > 0.2, "%.2f to %.2f" % (min(lens), max(lens)))
R = sm._rotation(np.random.default_rng(1), 12.0)
expect("the sensor-tilt rotation is a real rotation (orthonormal, determinant 1), a few degrees at most", np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-9 and np.degrees(np.arccos((np.trace(R) - 1) / 2)) < 25)
a1 = sm.augment(x0, np.random.default_rng(3))
expect("an augmented trial is float32 with 6 channels, and differs from the original", a1.dtype == np.float32 and a1.shape[1] == 6 and not np.allclose(a1[:len(x0)], x0[:len(a1)]))
expect("the same seed gives the same augmentation", np.allclose(sm.augment(x0, np.random.default_rng(3)), a1))

print("11. training on synthetic trials (a known right answer)")
train, test = make_set(11, n_walk=40, n_neg=16), make_set(12, n_walk=16, n_neg=8)
net1, h1 = sm.fit(train, epochs=60, seed=0)                      # 60 epochs is the default the real command uses
_, h2a = sm.fit(train[:16], epochs=6, seed=3)
_, h2b = sm.fit(train[:16], epochs=6, seed=3)
expect("the training loss falls to under a tenth of where it started", h1[-1] < h1[0] / 10, "%.3f to %.3f" % (h1[0], h1[-1]))
expect("and it did NOT collapse to counting nothing: the average walk gets counted as roughly the right size (the trivial answer, 0 steps, is what the first loss function fell into)", np.mean([sm.count_of(net1, t.x) for t in test if t.steps > 0]) > 0.6 * np.mean([t.steps for t in test if t.steps > 0]))
expect("the same seed gives the same training, epoch by epoch", np.allclose(h2a, h2b) and len(h2a) == 6)
cnt = np.array([sm.count_of(net1, t.x) for t in test]); tru = np.array([t.steps for t in test], dtype=float)
m = ev.metrics(tru, cnt)
zero = tru == 0
expect("on synthetic trials it never saw, the average error is a small fraction of the steps (under 10%)", m["mape"] < 10, "%.1f%%, %.2f steps" % (m["mape"], m["mae"]))
expect("and it finds almost no steps when there are none (the average count on 0-step trials is under 1.5)", float(cnt[zero].mean()) < 1.5, "%.2f" % float(cnt[zero].mean()))
dns = sm.density_of(net1, test[0].x)
expect("the density it draws is one number per reading, adding up to the count", dns.shape == (len(test[0].x),) and abs(dns.sum() - sm.count_of(net1, test[0].x)) < 1e-9)
exp_np = net1.to_numpy({"x": 1})
expect("exported to NumPy it counts the same as PyTorch, to a thousandth of a step", all(abs(exp_np.count(sd.features(t.x)) - sm.count_of(net1, t.x)) < 1e-3 for t in test[:6]))

print("11b. a run that gets stuck counting nothing is spotted and restarted")
import types
calls = []
real_train = sm._train_once
def fake_train(trials, epochs, seed, lr, batch, aug, width, log, make_net=None):
    calls.append(seed)
    net = sm.StepNet()
    return net, ([1.1] * epochs if len(calls) == 1 else [0.05] * epochs)            # the first attempt ends stuck, the second is healthy
real_collapsed = sm.is_collapsed
verdicts = iter([True, False])
sm._train_once, sm.is_collapsed = fake_train, (lambda net, trials: next(verdicts))
net_r, hist_r = sm.fit(train[:6], epochs=3, seed=5)
sm._train_once, sm.is_collapsed = real_train, real_collapsed
expect("a first attempt that ends stuck is thrown away, and another seed is tried", len(calls) == 2 and calls[0] == 5 and calls[1] == 1005, str(calls))
expect("the network records that it needed 1 restart and is not marked as collapsed", net_r.restarts == 1 and net_r.collapsed is False and hist_r[-1] == 0.05)
calls.clear()
sm._train_once, sm.is_collapsed = (lambda *a, **k: (calls.append(1) or (sm.StepNet(), [1.1] * 3))), (lambda net, trials: True)
net_c, _ = sm.fit(train[:6], epochs=3, seed=5, max_restarts=2)
sm._train_once, sm.is_collapsed = real_train, real_collapsed
expect("if every attempt gets stuck it gives up after 3 tries and says so, instead of looping forever", len(calls) == 3 and net_c.restarts == 2 and net_c.collapsed is True, str(len(calls)))
dead = sm.StepNet(); torch.nn.init.constant_(dead.head.bias, -25.0)
expect("a network that outputs nothing is recognised as collapsed; the trained network of section 11 is not; and with no walking trials there is nothing to count, so it is not", sm.is_collapsed(dead, train) and not sm.is_collapsed(net1, train) and not sm.is_collapsed(dead, [t for t in train if t.steps == 0]))
expect("the 60-epoch training in section 11 was not restarted (its loss began at %.2f and fell to %.2f)" % (h1[0], h1[-1]), net1.restarts == 0 and net1.collapsed is False)

print("11c. the architecture comparison")
import argparse, contextlib, io as _io
import arch_compare as ac, steps_alt as alt
d_true = np.array([10.0, 10, 10, 10])
m, lo, hi = ac.paired_ci(d_true, [10, 10, 12, 12], [11, 11, 11, 11])
expect("paired difference: errors (0, 0, 2, 2) against (1, 1, 1, 1) average out to 0, and the interval brackets 0", abs(m) < 1e-9 and lo <= 0 <= hi, "%.2f (%.2f to %.2f)" % (m, lo, hi))
m, lo, hi = ac.paired_ci(d_true, [10, 10, 10, 10], [12, 12, 12, 12])
expect("a variant that is exactly right against one that is 2 out everywhere: mean difference -2, interval -2 to -2 (negative means better)", abs(m + 2) < 1e-9 and abs(lo + 2) < 1e-9 and abs(hi + 2) < 1e-9)
expect("every variant is built, gives one density per reading that is never negative, and starts at about the same total density (so the comparison starts level)",
       all(tuple(b().eval()(torch.randn(1, 8, 400)).shape) == (1, 400) and bool((b().eval()(torch.randn(1, 8, 400)) >= 0).all()) and 0.15 < float(b().eval()(torch.randn(1, 8, 400)).sum()) < 1.0 for _, b, _ in alt.VARIANTS.values()))
expect("the shipped network is the reference, listed first, and the variants cover width, depth, kernel size and a recurrent network", list(alt.VARIANTS)[0] == "reference" and {"narrow", "wide", "shallow", "deep", "kernel3", "kernel9", "gru"} <= set(alt.VARIANTS))
expect("the reach of kernel 5 with dilations 1 to 16 is 125 readings, as for the shipped network", alt.receptive_field(5, (1, 2, 4, 8, 16)) == 125)
fake = {"reference": {"what": "ref", "params": 1, "restarts": [0], "metrics": {"mae": 0.7}, "mae_ci95": (0.5, 0.9), "mae_walking": 0.9, "vs_reference": None, "beats_reference_by_rule": False},
        "better": {"what": "b", "params": 2, "restarts": [0], "metrics": {"mae": 0.4}, "mae_ci95": (0.3, 0.5), "mae_walking": 0.5, "vs_reference": {"mean": -0.3, "lo": -0.5, "hi": -0.1}, "beats_reference_by_rule": True}}
ns = argparse.Namespace(synthetic=False, epochs=60)
ac.write_summary(tmp / "cmp_win.md", fake, 55, 5, ns)
fake["better"].update({"vs_reference": {"mean": -0.06, "lo": -0.3, "hi": 0.2}, "beats_reference_by_rule": False})
ac.write_summary(tmp / "cmp_none.md", fake, 55, 5, ns)
expect("when a variant is better by the rule the summary says so, by name", "By the rule, better beat the shipped network" in (tmp / "cmp_win.md").read_text() and "beats it by the rule" in (tmp / "cmp_win.md").read_text())
expect("when none is, the summary says the shipped network stays and why (the interval includes zero)", "No variant beat the shipped network by the rule" in (tmp / "cmp_none.md").read_text() and "cannot be told apart from noise" in (tmp / "cmp_none.md").read_text())
expect("the rule is written into the summary, with its margin of 0.15 steps", "at least 0.15 steps" in (tmp / "cmp_none.md").read_text() and "fixed before any result was seen" in (tmp / "cmp_none.md").read_text())
cmp_out = tmp / "cmp_run"
args = argparse.Namespace(trials=None, synthetic=True, legacy=None, out=str(cmp_out), epochs=2, folds=2, seed=0, variants=["narrow"], resume=False)
with contextlib.redirect_stdout(_io.StringIO()): first = ac.run(args)
args2 = argparse.Namespace(**{**vars(args), "variants": ["narrow", "shallow"], "resume": True})
buf = _io.StringIO()
with contextlib.redirect_stdout(buf): second = ac.run(args2)
expect("an interrupted run can be resumed: the second run says what it skipped, and only trains the new variant", "resuming: already done: reference, narrow" in buf.getvalue() and buf.getvalue().count("weights") == 1 and "shallow" in buf.getvalue(), buf.getvalue()[:160].replace("\n", " | "))
expect("the skipped variants keep exactly their earlier predictions (they were not retrained)", second["narrow"]["pred"] == first["narrow"]["pred"] and second["reference"]["pred"] == first["reference"]["pred"])
expect("and the final files cover all three variants, with the paired differences against the reference", set(second) == {"reference", "narrow", "shallow"} and second["narrow"]["vs_reference"] is not None and (cmp_out / "architecture_comparison.md").exists())
args3 = argparse.Namespace(**{**vars(args), "epochs": 3, "variants": ["narrow"], "resume": True})
buf = _io.StringIO()
with contextlib.redirect_stdout(buf): ac.run(args3)
expect("a saved run with different settings (epochs) is NOT reused", "resuming" not in buf.getvalue())

print("12. the whole command, end to end")
out = tmp / "run"
cmd = [sys.executable, str(ROOT / "train_steps.py"), "--synthetic", "--quick", "--out", str(out / "results"), "--models", str(out / "models")]
res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
expect("it runs to the end", res.returncode == 0, (res.stderr or res.stdout)[-200:].replace("\n", " "))
files = sorted(p.name for p in (out / "results").glob("*")) if (out / "results").exists() else []
expect("it writes the figures, results.json and summary.md", all(n in files for n in ("scatter_network.png", "errors_network.png", "network_vs_baseline.png", "example_typical.png", "example_worst.png", "training_loss.png", "results.json", "summary.md")), str(files))
expect("it writes the network as NumPy-readable JSON, and the PyTorch weights", (out / "models" / "step_net.json").exists() and (out / "models" / "step_net.pt").exists())
if (out / "results" / "results.json").exists():
    R = json.loads((out / "results" / "results.json").read_text(encoding="utf-8"))
    expect("results.json holds the network and baseline errors with 95% intervals, and one out-of-fold row per trial", R["network_mae_ci95"][0] <= R["network"]["mae"] <= R["network_mae_ci95"][1] and len(R["oof"]) == R["n_trials"] and "baseline" in R)
    expect("the NumPy copy matched PyTorch inside the command too", R["numpy_vs_torch_max_diff"] < 1e-3, "%.2e" % R["numpy_vs_torch_max_diff"])
    summary = (out / "results" / "summary.md").read_text(encoding="utf-8")
    expect("the summary says in capitals that synthetic numbers are not real accuracy", "made-up trials" in summary and "say nothing about real accuracy" in summary)
    expect("and states the limits: your own count is the truth, the number of trials, the sensor placement", "Limits to say out loud" in summary and "your own count" in summary and "same way on the same ankle" in summary)
    loaded = NumpyStepNet.load(out / "models" / "step_net.json")
    expect("the saved model records how it was made", loaded.meta.get("synthetic") is True and loaded.meta.get("epochs") == 30 and loaded.meta.get("trials", 0) >= 8)
    expect("results.json compares the two methods on the trials WITH steps only (the no-step trials flatter whichever method is quiet)", R["walking_only"]["n"] > 0 and "baseline" in R["walking_only"] and "trials WITH steps only" in summary)
    expect("and gives a distance model for each person from that person's own walks, and says why", "synth" in R["distance"]["per_person"] and R["distance"]["per_person"]["synth"]["n"] >= 6 and "different step lengths" in summary, str(list(R["distance"]["per_person"])))
    expect("results.json says how many networks needed a restart, and the summary says it in words", len(R["cv_restarts"]) >= 2 and "needed a restart with another seed" in summary, str(R["cv_restarts"]))
res2 = subprocess.run([sys.executable, str(ROOT / "train_steps.py")], cwd=ROOT, capture_output=True, text=True)
expect("run without --trials it explains what to give it", res2.returncode != 0 and "--trials" in (res2.stderr + res2.stdout))

print()
print("ALL PASSED" if problems == 0 else f"{problems} CHECK(S) FAILED")
sys.exit(1 if problems else 0)
