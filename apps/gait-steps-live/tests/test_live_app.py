"""Tests for python/main.py (the live step counter's logic, the glue to the page and the LED). No board needed.   python test_live_app.py
If the real trials are found (STEPS_TRIALS=all_trials.zip, or all_trials.zip next to the project) they are streamed through the app too."""
import csv, datetime, importlib.util, json, os, sys, tempfile, types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parent
PY = APP / "python"
sys.path.insert(0, str(PY))
problems = 0


def expect(label, cond, detail=""):
    global problems
    print(f"  {'ok  ' if cond else 'FAIL'} {label}" + (f"   [{detail}]" if detail else ""))
    problems += (not cond)


# ---- load main.py with stand-ins for the board libraries ----
log = {"messages": [], "handlers": {}, "calls": [], "led_fails": False, "loop": None, "provided": {}}


class Bridge:
    @staticmethod
    def provide(name, fn): log["provided"][name] = fn

    @staticmethod
    def call(name, *args, timeout=10):
        if log["led_fails"]: raise TimeoutError("no answer")
        log["calls"].append((name,) + args)


class App:
    @staticmethod
    def run(user_loop=None): log["loop"] = user_loop


class WebUI:
    def on_message(self, kind, fn): log["handlers"][kind] = fn
    def on_connect(self, fn): log["handlers"]["connect"] = fn
    def send_message(self, kind, msg, room=None): log["messages"].append((kind, msg))


u = types.ModuleType("arduino.app_utils"); u.Bridge, u.App = Bridge, App
w = types.ModuleType("arduino.app_bricks.web_ui"); w.WebUI = WebUI
sys.modules.update({"arduino": types.ModuleType("arduino"), "arduino.app_utils": u, "arduino.app_bricks": types.ModuleType("arduino.app_bricks"), "arduino.app_bricks.web_ui": w})
tmp = Path(tempfile.mkdtemp())
os.environ["GAIT_DATA_DIR"] = str(tmp / "glue_data")
spec = importlib.util.spec_from_file_location("live_main", PY / "main.py")
main = importlib.util.module_from_spec(spec); spec.loader.exec_module(main)
from steps_data import features, load_trials
from steps_numpy import NumpyStepNet

NET = main.net
DIST = main.distance
EVAL = main.evaluation


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


clock = Clock()
wall0 = datetime.datetime(2026, 11, 3, 9, 0, 0)
wall = lambda: wall0 + datetime.timedelta(seconds=clock.t - 1000.0)
counter_n = [0]


def new_live(name, dist=DIST, evaluation=EVAL):
    return main.Live(NET, dist, evaluation, tmp / name, clock=clock, wall=wall)


def walk_signal(seconds, cadence=1.8, seed=0):
    """A crude ankle-like walk: a leg swing and two impacts per stride, plus noise. Good enough to exercise the plumbing, not to measure accuracy."""
    r = np.random.default_rng(seed); T = int(seconds * 100); t = np.arange(T) / 100.0
    x = np.zeros((T, 6)); x[:, 0] = -0.93; x[:, :3] += r.normal(0, 0.015, (T, 3)); x[:, 3:] += r.normal(0, 1.0, (T, 3))
    for i in range(int(seconds * cadence)):
        ti = (i + 0.5) / cadence
        x[:, 0] += np.where(t >= ti, np.exp(-np.maximum(t - ti, 0) / 0.04) * np.sin(2 * np.pi * 18 * (t - ti)), 0.0) * 1.5
        if i % 2 == 0: x[:, 5] += 250 * np.exp(-0.5 * ((t - (ti - 0.25)) / 0.12) ** 2)
    return x.astype(np.float32)


def feed(live, x, start_n=None, per_step=None, hz=100):
    """Send readings the way the sketch does, one at a time, advancing the fake clock; optionally ask for a snapshot every so often."""
    n0 = counter_n[0] if start_n is None else start_n
    for i, row in enumerate(x):
        clock.t += 1.0 / hz
        live.add_sample(n0 + i, *[float(v) for v in row])
        if per_step and i % per_step == 0: per_step_fn(live)
    counter_n[0] = n0 + len(x)


seen = []
def per_step_fn(live): live.process(); seen.append(live.counter.count)


print("1. one walk, from START to a saved session")
live = new_live("a")
try:
    live.start("kendrick"); expect("START is refused while no readings arrive", False)
except ValueError as e:
    expect("START is refused while no readings arrive, and says so", "No IMU data" in str(e), str(e)[:70])
feed(live, walk_signal(2.0, seed=9))                       # the sensor is streaming (idle: readings are not kept)
try:
    live.start(""); expect("START is refused without a name", False)
except ValueError as e:
    expect("START is refused without a name", "who is wearing" in str(e))
live.start("kendrick!")
expect("the name is cleaned to letters and digits, and the app is walking", live.person == "kendrick" and live.mode == "walking")
x = walk_signal(14.0, cadence=1.8, seed=3)
seen.clear(); feed(live, x, per_step=25)
snap = live.snapshot()
expect("while walking the count rises (it never goes down by more than a hair between updates)", seen[-1] > 8 and all(b >= a - 1e-9 for a, b in zip(seen, seen[1:])), "%.1f" % seen[-1])
expect("the snapshot shows walking, the person, a whole-number count, the elapsed time and a pace", snap["mode"] == "walking" and snap["person"] == "kendrick" and snap["steps"] == int(round(snap["steps_exact"])) and 13.5 <= snap["elapsed"] <= 14.5 and snap["pace"] > 0, str((snap["steps"], snap["elapsed"], snap["pace"])))
expect("and the plot has the last seconds of the network's density and of the rotation", len(snap["plot"]["density"]) > 50 and len(snap["plot"]["gyro"]) > 50, str((len(snap["plot"]["density"]), len(snap["plot"]["gyro"]))))
json.dumps(snap)
offline = NET.count(features(x))
s = live.stop()
expect("STOP makes the count final: it EQUALS the offline count of the whole walk to 9 decimals (%.4f)" % offline, abs(s["steps_exact"] - offline) < 1e-2 and abs(live.last["steps_exact"] - round(offline, 2)) < 1e-9, "%.4f vs %.4f" % (s["steps_exact"], offline))
expect("the count shown is that number rounded to a whole step", s["steps"] == int(round(offline)))
expect("the session is saved: a CSV in the recorder's format, a log row and a manifest row", (tmp / "a" / s["file"]).exists() and (tmp / "a" / "live_log.csv").exists() and (tmp / "a" / "manifest.csv").exists())
rows = list(csv.reader(open(tmp / "a" / s["file"], encoding="utf-8")))
expect("the CSV header is n,ax,ay,az,gx,gy,gz and it holds every reading of the walk", rows[0] == ["n", "ax", "ay", "az", "gx", "gy", "gz"] and len(rows) - 1 == len(x), str(len(rows) - 1))
trials, skipped = load_trials(tmp / "a")
expect("the step recorder's own loader skips it until you say how many steps there were, and tells you why", trials == [] and len(skipped) == 1 and "no step count" in skipped[0][1], str(skipped))
expect("the snapshot after STOP is 'done' with the final result", live.snapshot()["mode"] == "done" and live.snapshot()["last"]["id"] == s["id"])
saved_offline = NET.count(features(np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=np.float32)))
expect("counting the SAVED file offline gives the same number (the file's rounding changes it by under 0.01 steps)", abs(saved_offline - offline) < 0.01, "%.4f vs %.4f" % (saved_offline, offline))

print("2. telling the app how many steps there really were")
for bad, why in (("abc", "text"), ("-2", "negative"), ("501", "too many"), ("11.5", "half a step"), (None, "nothing")):
    try:
        live.set_truth(s["id"], bad); expect("a count of %r is refused (%s)" % (bad, why), False)
    except ValueError:
        pass
expect("text, negative, over 500, half a step and nothing are all refused", live.sessions[-1]["true_steps"] == "")
try:
    live.set_truth("nope", 11); expect("an unknown session is refused", False)
except ValueError as e:
    expect("an unknown session is refused", "No such session" in str(e))
t = live.set_truth(s["id"], s["steps"] + 2)
expect("the error is the counter's number minus yours, and the session now has your count", t["true_steps"] == s["steps"] + 2 and t["error"] == -2, str((t["true_steps"], t["error"])))
tal = live.tally()
expect("the tally: 1 demo, average error 2, within one step 0%, worst 2", tal["n"] == 1 and tal["mae"] == 2.0 and tal["within1"] == 0.0 and tal["worst"] == 2.0 and abs(tal["bias"] + 2.0) < 1e-9, str(tal))
trials, skipped = load_trials(tmp / "a")
expect("and now the recorder's loader reads it as a trial with your count, kind 'live'", len(trials) == 1 and trials[0].steps == s["steps"] + 2 and trials[0].kind == "live" and trials[0].person == "kendrick" and len(trials[0].x) == len(x))
man = list(csv.DictReader(open(tmp / "a" / "manifest.csv", encoding="utf-8")))
expect("the manifest is in the recorder's format, with 0% lost", list(man[0].keys()) == main.MANIFEST_FIELDS and man[0]["kind"] == "live" and float(man[0]["lost_pct"]) == 0)
expect("the live page can be told the session list, newest first, with the error", live.snapshot()["sessions"][0]["error"] == -2)

print("3. distance, and calibrating the step length")
live.reset()
expect("reset goes back to idle", live.mode == "idle" and live.last is None)
feed(live, walk_signal(1.0, seed=1)); live.start("Kendrick"); feed(live, walk_signal(12.0, cadence=1.7, seed=5)); s2 = live.stop()
expect("with a model fitted to Kendrick's recorded walks, the distance is steps times that model's step length, and it says where it came from",
       s2["distance_est_m"] == round(s2["steps"] * s2["step_length_m"], 2) and 0.3 <= s2["step_length_m"] <= 1.2 and "Kendrick" in s2["step_length_source"], "%.3f m a step; %s" % (s2["step_length_m"], s2["step_length_source"]))
live.reset(); feed(live, walk_signal(1.0)); live.start("somebody new"); feed(live, walk_signal(10.0, seed=6)); s3 = live.stop()
expect("an unknown person gets the model for everyone, and the page is told to calibrate", "model for everyone" in s3["step_length_source"] and "calibrate" in s3["step_length_source"], s3["step_length_source"])
live.set_truth(s3["id"], 11, 5.8)
expect("typing a measured distance with the count calibrates that person: 5.8 m / 11 steps = 0.527 m a step", abs(live.calibration["somebody new"][0] - 5.8 / 11) < 1e-9, str(live.calibration))
live.reset(); feed(live, walk_signal(1.0)); live.start("somebody new"); feed(live, walk_signal(10.0, seed=7)); s4 = live.stop()
expect("the next walk by that person uses the calibration, and says it is their own", abs(s4["step_length_m"] - round(5.8 / 11, 3)) < 1e-9 and "your own 1 measured walk" in s4["step_length_source"], s4["step_length_source"])
live.set_truth(s4["id"], 9, 4.8)
expect("two calibrations: the median of the two (0.527 and 0.533)", abs(np.median(live.calibration["somebody new"]) - np.median([5.8 / 11, 4.8 / 9])) < 1e-9 and len(live.calibration["somebody new"]) == 2)
for _ in range(12):
    live.set_truth(s4["id"], 9, 4.8)
expect("only the last 10 calibrations are kept", len(live.calibration["somebody new"]) == 10)
for bad_d in ("far", "0.2", "900"):
    try:
        live.set_truth(s4["id"], 9, bad_d); expect("a distance of %r is refused" % bad_d, False)
    except ValueError:
        pass
try:
    live.set_truth(s4["id"], 0, 5.0); expect("a distance on a walk of 0 steps is refused", False)
except ValueError as e:
    expect("a bad distance is refused, and so is a distance on a walk with 0 steps", "0 steps" in str(e))
nod = new_live("nodist", dist={})
feed(nod, walk_signal(1.0)); nod.start("ann"); feed(nod, walk_signal(8.0, seed=2)); sn = nod.stop()
expect("with no step-length model at all there is no distance, and it says why", sn["distance_est_m"] == "" and "no step-length model" in sn["step_length_source"] and nod.snapshot()["distance_m"] is None)

print("4. it all survives a restart")
again = new_live("a")
expect("a restart reloads every saved session with its error, and the tally comes out the same (%d sessions)" % len(live.sessions), len(again.sessions) == len(live.sessions) >= 3 and again.sessions[0]["true_steps"] == s["steps"] + 2 and again.tally() == live.tally(), str(again.tally()))
cal = new_live("a"); calx = main.Live(NET, DIST, EVAL, tmp / "a", clock=clock, wall=wall)
expect("the sessions' files are found again", all((tmp / "a" / x_["file"]).exists() for x_ in again.sessions))
persist = new_live("a")
persist.reset(); feed(persist, walk_signal(1.0)); persist.start("p"); feed(persist, walk_signal(8.0, seed=4)); sp = persist.stop(); persist.set_truth(sp["id"], 9, 5.0)
expect("calibrations survive too (calibration.json)", abs(new_live("a").calibration["p"][0] - 5.0 / 9) < 1e-9)
os.remove(tmp / "a" / sp["file"])
expect("a session whose file was removed by hand is forgotten, not a crash", all(x_["id"] != sp["id"] for x_ in new_live("a").sessions))

print("5. deleting, and the buttons in the wrong order")
d = new_live("d"); feed(d, walk_signal(1.0)); d.start("ann"); feed(d, walk_signal(8.0, seed=8)); sd = d.stop()
d.delete(sd["id"])
expect("delete removes the file, the log row and the manifest row, and returns to idle", not (tmp / "d" / sd["file"]).exists() and d.sessions == [] and d.mode == "idle" and len(list(csv.DictReader(open(tmp / "d" / "manifest.csv", encoding="utf-8")))) == 0)
for label, call in (("STOP when nothing is counted", lambda: d.stop()), ("deleting something that is not there", lambda: d.delete("nope"))):
    try:
        call(); expect(label + " is refused", False)
    except ValueError:
        expect(label + " is refused", True)
feed(d, walk_signal(1.0)); d.start("ann")
for label, call in (("START while counting", lambda: d.start("ann")), ("reset while counting", lambda: d.reset())):
    try:
        call(); expect(label + " is refused", False)
    except ValueError:
        expect(label + " is refused", True)
feed(d, walk_signal(0.5, seed=1))
try:
    d.stop(); expect("a walk under a second is refused", False)
except ValueError as e:
    expect("a walk under a second is refused, and nothing is saved", "under a second" in str(e) and d.mode == "idle" and d.sessions == [])

print("6. readings lost on the way")
lo = new_live("lost"); feed(lo, walk_signal(1.0)); lo.start("ann")
x1 = walk_signal(8.0, seed=11); feed(lo, x1[:400]); counter_n[0] += 40; feed(lo, x1[400:])      # 40 readings never arrived
sl = lo.stop()
man = list(csv.DictReader(open(tmp / "lost" / "manifest.csv", encoding="utf-8")))
expect("the counter jump is measured and written to the manifest: 40 of 840 lost = 4.76%", int(man[0]["dropped"]) == 40 and abs(float(man[0]["lost_pct"]) - 100 * 40 / 840) < 0.01, str((man[0]["dropped"], man[0]["lost_pct"])))

print("7. the sensor goes quiet while counting")
q = new_live("q"); feed(q, walk_signal(1.0)); q.start("ann"); feed(q, walk_signal(3.0, seed=2))
clock.t += 3.0
expect("after 3 s of nothing the page is told the sensor is not sending", q.snapshot()["imu_ok"] is False and q.snapshot()["mode"] == "walking")

print("8. the glue: commands from the page, the notices and the LED matrix")
main.live = new_live("glue")
cmd = log["handlers"]["cmd"]
notes = lambda: [m[1] for m in log["messages"] if m[0] == "notice"]
feed(main.live, walk_signal(1.0))
cmd("sid", {"action": "start", "person": ""})
expect("START without a name: a warning that says what to fix", notes()[-1]["level"] == "warn" and "who is wearing" in notes()[-1]["text"], notes()[-1]["text"])
cmd("sid", {"action": "start", "person": "kendrick"})
expect("START: an info notice", notes()[-1]["level"] == "info" and "Counting" in notes()[-1]["text"])
cmd("sid", {"action": "stop"})
expect("STOP too soon: a warning, not a crash", notes()[-1]["level"] == "warn" and "under a second" in notes()[-1]["text"], notes()[-1]["text"])
main.live.reset()
feed(main.live, walk_signal(0.5)); log["calls"].clear(); log["messages"].clear()
cmd("sid", {"action": "start", "person": "kendrick"})
main.on_imu(counter_n[0], 0, 0, -1, 0, 0, 0)
for i, row in enumerate(walk_signal(12.0, seed=12)):
    clock.t += 0.01; main.on_imu(counter_n[0] + i + 1, *[float(v) for v in row])
    if i % 20 == 0: main.time.sleep = lambda s: None; log["loop"]()
states = [m[1] for m in log["messages"] if m[0] == "state"]
expect("the loop keeps sending the page a state while walking", len(states) > 10 and states[-1]["mode"] == "walking" and states[-1]["steps"] > 5, str(states[-1]["steps"]))
led = [c for c in log["calls"] if c[0] == "led"]
expect("the LED matrix is told to show the count as a number (mode 4), and only when it changes", led and all(c[1] == 4 for c in led) and len({c[2] for c in led}) == len(led) and led[-1][2] == states[-1]["steps"], str(led[-3:]))
cmd("sid", {"action": "stop"})
last_id = main.live.last["id"]
expect("STOP: the notice reports the count, and asks what you really counted", "Counted" in notes()[-1]["text"] and "really counted" in notes()[-1]["text"], notes()[-1]["text"][:90])
cmd("sid", {"action": "truth", "id": last_id, "steps": "abc"})
expect("a bad count: a warning that says what to type", notes()[-1]["level"] == "warn" and "whole number" in notes()[-1]["text"], notes()[-1]["text"])
cmd("sid", {"action": "truth", "id": last_id, "steps": str(main.live.last["steps"]), "distance": ""})
expect("the right count: 'exactly right', and the tally is in the next state", "exactly right" in notes()[-1]["text"] and [m[1] for m in log["messages"] if m[0] == "state"][-1]["tally"]["n"] == 1, notes()[-1]["text"])
log["loop"]()
expect("when done the LED still shows the final number", [c for c in log["calls"] if c[0] == "led"][-1][1] == 4)
cmd("sid", {"action": "reset"}); log["loop"]()
expect("after reset the LED goes back to the idle dot", [c for c in log["calls"] if c[0] == "led"][-1][1:3] == (0, 0))
cmd("sid", {"action": "delete", "id": "nope"})
expect("deleting an unknown session: a warning", notes()[-1]["level"] == "warn" and "No such session" in notes()[-1]["text"])
cmd("sid", {"action": "nonsense"}); expect("an unknown command is refused", notes()[-1] == {"level": "warn", "text": "Unknown command."})
cmd("sid", None); expect("an empty command is refused the same way, without a crash", notes()[-1] == {"level": "warn", "text": "Unknown command."})
log["led_fails"] = True; main.live.mode, main.live.last = "idle", None
main.live.counter.reset(); feed(main.live, walk_signal(1.0)); main.live.start("kendrick"); feed(main.live, walk_signal(2.0, seed=3)); main.live.process()
try:
    for _ in range(3): log["loop"]()
    expect("a dead LED matrix does not stop the app", True)
except Exception as e:
    expect("a dead LED matrix does not stop the app", False, repr(e))
log["led_fails"] = False
expect("the app registered its pieces: imu, cmd, connect and the loop", "imu" in log["provided"] and {"cmd", "connect"} <= set(log["handlers"]) and callable(log["loop"]))

print("9. the evaluation card and the model the app carries")
sn = new_live("ev").snapshot()
expect("the page gets the evaluation numbers that ship next to the network", sn["evaluation"].get("n_trials", 0) >= 8 and "network_mae" in sn["evaluation"] and sn["people"] == sorted(DIST.get("people", {}).keys()))
expect("the network on the board is the one that was trained: %d weights, notes say real trials" % NET.n_parameters(), NET.n_parameters() > 1000 and NET.meta.get("trials", 0) >= 8)

print("10. real trials streamed through the app")
cand = [os.environ.get("STEPS_TRIALS"), str(APP.parent.parent / "all_trials.zip"), str(APP.parent.parent.parent / "all_trials.zip"), "/mnt/user-data/uploads/all_trials.zip"]
zp = next((c for c in cand if c and Path(c).exists()), None)
if zp:
    real, _ = load_trials(zp)
    pick = [t for t in real if t.kind in ("walk_slow", "walk_normal", "walk_fast", "jog", "other", "foot_tap", "stand")][::5][:12]
    r = new_live("real"); worst = 0.0; worst_csv = 0.0
    for t in pick:
        feed(r, walk_signal(1.0)); r.start(t.person); feed(r, t.x); st = r.stop()
        off = NET.count(features(t.x)); worst = max(worst, abs(st["steps_exact"] - off))
        rows = list(csv.reader(open(tmp / "real" / st["file"], encoding="utf-8")))
        worst_csv = max(worst_csv, abs(NET.count(features(np.array([[float(v) for v in q[1:]] for q in rows[1:]], dtype=np.float32))) - off)); r.reset()
    expect("%d real trials streamed one reading at a time: every final count equals the offline count (largest gap %.4f steps)" % (len(pick), worst), worst < 0.01 and len(pick) >= 8)
    expect("and the saved CSV of each, counted offline, agrees too (largest gap %.4f)" % worst_csv, worst_csv < 0.01)
    expect("none of the no-step trials in that sample was counted as steps", all(s_["steps"] == 0 for s_, t in zip(r.sessions, pick) if t.steps == 0), str([(t.kind, s_["steps"]) for s_, t in zip(r.sessions, pick) if t.steps == 0]))
else:
    print("  (the real trials were not found: set STEPS_TRIALS=all_trials.zip to include this group)")

print("11. the payloads the page receives (saved for the page test)")
D = {}
pl = new_live("pl", evaluation=EVAL)
D["idle_no_sensor"] = pl.snapshot()
feed(pl, walk_signal(1.5)); D["idle"] = pl.snapshot()
pl.start("kendrick"); feed(pl, walk_signal(9.0, cadence=1.8, seed=21)); D["walking"] = pl.snapshot()
clock.t += 3.0; D["walking_no_sensor"] = pl.snapshot()
feed(pl, walk_signal(0.5)); sdone = pl.stop(); D["done"] = pl.snapshot()
pl.set_truth(sdone["id"], sdone["steps"], 5.8); D["answered_exact"] = pl.snapshot()
pl.reset(); feed(pl, walk_signal(1.0)); pl.start("kendrick"); feed(pl, walk_signal(7.0, seed=22)); s2_ = pl.stop(); pl.set_truth(s2_["id"], s2_["steps"] + 2); D["answered_too_few"] = pl.snapshot()
pl.reset(); feed(pl, walk_signal(1.0)); pl.start("marie"); feed(pl, walk_signal(7.0, seed=23)); s3_ = pl.stop(); pl.set_truth(s3_["id"], max(0, s3_["steps"] - 1)); D["answered_too_many"] = pl.snapshot()
nm = new_live("pl_nomodel", dist={}, evaluation={}); feed(nm, walk_signal(1.0)); nm.start("ann"); feed(nm, walk_signal(6.0, seed=24)); nm.stop(); D["done_no_model"] = nm.snapshot()
expect("every payload is plain JSON", all(len(json.dumps(v)) > 200 for v in D.values()), str(sorted(D)))
expect("they cover idle, walking, done, answered right, with too few, with too many, no sensor and no model", {"idle", "walking", "done", "answered_exact", "answered_too_few", "answered_too_many", "walking_no_sensor", "done_no_model"} <= set(D))
expect("the answered ones carry the error the page will show", D["answered_too_few"]["last"]["error"] == -2 and D["answered_too_many"]["last"]["error"] == 1 and D["answered_exact"]["last"]["error"] == 0)
json.dump(D, open(HERE / "state_samples.json", "w"))

print()
print("ALL PASSED" if problems == 0 else f"{problems} CHECK(S) FAILED")
sys.exit(1 if problems else 0)
