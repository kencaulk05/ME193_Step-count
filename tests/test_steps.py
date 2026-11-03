"""Tests for the step counter (steps_*.py, train_steps.py). No board needed.   python tests/test_steps.py
The synthetic trials in these tests have a known right answer by construction: they check the MACHINERY, not real accuracy."""
import csv, io, json, os, subprocess, sys, tempfile, zipfile
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
expect("two usable trials are loaded; the other three are skipped, each with its reason", [t.name for t in trials] == ["a.csv", "b.csv"] and len(skipped) == 3, str(skipped))
reasons = dict(skipped)
expect("the reasons are in plain words: Edit for a missing count, lost readings, a missing file", "Edit" in reasons["c.csv"] and "lost" in reasons["d.csv"] and "missing" in reasons["gone.csv"], str(reasons))
t = trials[0]
expect("a trial carries person, kind, steps, distance and a (T, 6) float32 signal", (t.person, t.kind, t.steps, t.distance_m) == ("ann", "walk_normal", 20, 10.0) and t.x.shape == (500, 6) and t.x.dtype == np.float32 and abs(t.seconds - 5.0) < 1e-9)
expect("no distance means NaN, and a 0-step trial is kept", np.isnan(trials[1].distance_m) and trials[1].steps == 0)
(tmp / "folder").mkdir()
with zipfile.ZipFile(tmp / "t.zip") as z: z.extractall(tmp / "folder")
t2, _ = sd.load_trials(tmp / "folder")
expect("a plain folder works the same as the zip", [x.name for x in t2] == ["a.csv", "b.csv"] and np.allclose(t2[0].x, trials[0].x))
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
expect("trials still waiting for their count can be loaded on request (steps -1)", [x.steps for x in ul if x.name == "c.csv"] == [-1])

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
    walk = sd._read_csv_text(next(LEGACY.glob("Kendrick__walk__*.csv")).read_text(encoding="utf-8"), "walk")[200:-200]
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
def fake_train(trials, epochs, seed, lr, batch, aug, width, log):
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
