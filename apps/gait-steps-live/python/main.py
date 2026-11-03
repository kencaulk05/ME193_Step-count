"""How many steps? - the live counter.

The sketch sends an IMU reading 100 times a second:  imu(n, ax, ay, az, gx, gy, gz)  (n = counter, a in g, gyro in degrees/second).
This app runs the trained step network on that stream while you walk, shows the count on the web page (port 7000) and as a number on the
LED matrix, and keeps an honest record of how right it was:

    1. choose the person, press START, walk. The count goes up as you walk (the last 0.6 s of it is an estimate that can still change)
    2. press STOP. The count becomes final: exactly what the network says for the whole walk
    3. type the steps you really counted, and the distance if you measured it. Now the app knows how wrong it was
    4. every session is saved in the step recorder's format, so it can later be used as training data

Files go to  assets/data/  (the page serves that folder):
    <person>__live__<date>-<time>.csv   n, ax, ay, az, gx, gy, gz        manifest.csv   in the recorder's format (kind "live")
    live_log.csv                         one row per session: what it counted, what you counted, the error
    calibration.json                     each person's own measured step lengths

THE DISTANCE is steps times a step length. The step length comes from the first of these that exists: the person's own calibration (the
median of the walks where you typed a measured distance), the model fitted to that person's recorded trials, or the model for everyone.
"""

import collections
import csv
import datetime
import json
import math
import os
import re
import threading
import time
from pathlib import Path

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI

import numpy as np                     # after the board libraries, as in the apps that already ran on this board
from steps_data import FS, features
from steps_live import LiveCounter
from steps_numpy import NumpyStepNet

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
DATA_DIR = Path(os.environ.get("GAIT_DATA_DIR") or HERE.parent / "assets" / "data")
MAX_SECONDS = 15 * 60                  # a session ends by itself after this long
MIN_IMU_RATE = 50                      # readings in the last second that count as "the sensor is there"
MAX_STEPS = 500
LOG_FIELDS = ["id", "start", "person", "seconds", "steps", "steps_exact", "distance_est_m", "step_length_m", "true_steps", "error", "true_distance_m", "file"]
MANIFEST_FIELDS = ["file", "subject", "kind", "true_steps", "distance_m", "note", "start", "seconds", "samples", "dropped", "lost_pct"]
STEP_MIN, STEP_MAX = 0.3, 1.2          # metres: a step length outside this is nonsense


def clean_name(text):
    return re.sub(r"[^A-Za-z0-9_ -]", "", str(text or "")).strip()[:20]


class Live:
    """Everything the app knows, with no web page and no board in it, so it can be tested on its own."""

    def __init__(self, net, distance, evaluation, data_dir, clock=time.monotonic, wall=datetime.datetime.now):
        self.net, self.distance, self.evaluation = net, distance or {}, evaluation or {}
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clock, self.wall = clock, wall
        self.lock = threading.RLock()
        self.counter = LiveCounter(net)
        self.mode = "idle"                               # idle -> walking -> done
        self.person = ""
        self.pending = []                                # readings the sketch sent, waiting to be given to the counter
        self.samples = []                                # (n, ax, ay, az, gx, gy, gz) of this session
        self.arrivals = collections.deque()
        self.t_wall = None
        self.sessions = self._load_log()
        self.calibration = self._load_calibration()
        self.last = None

    # ---- the stream ----
    def add_sample(self, n, ax, ay, az, gx, gy, gz):
        now = self.clock()
        with self.lock:
            self.arrivals.append(now)
            while self.arrivals and now - self.arrivals[0] > 1.0:
                self.arrivals.popleft()
            if self.mode == "walking" and len(self.samples) + len(self.pending) < MAX_SECONDS * FS:
                self.pending.append((n, ax, ay, az, gx, gy, gz))

    def imu_rate(self):
        with self.lock:
            now = self.clock()
            while self.arrivals and now - self.arrivals[0] > 1.0:
                self.arrivals.popleft()
            return len(self.arrivals)

    def process(self):
        """Give the counter the readings that arrived since the last call (called about 5 times a second)."""
        with self.lock:
            if not self.pending:
                return
            rows, self.pending = self.pending, []
            self.samples.extend(rows)
            if self.mode == "walking":
                self.counter.push(np.array([r[1:] for r in rows], dtype=np.float32))

    # ---- the buttons ----
    def start(self, person):
        with self.lock:
            if self.mode == "walking":
                raise ValueError("Already counting: press STOP first.")
            person = clean_name(person)
            if not person:
                raise ValueError("Type who is wearing it (letters and digits).")
            if self.imu_rate() < MIN_IMU_RATE:
                raise ValueError("No IMU data is arriving (about %d a second). Check the sensor before starting." % self.imu_rate())
            self.counter.reset()
            self.pending, self.samples = [], []
            self.person, self.mode, self.t_wall, self.last = person, "walking", self.wall(), None

    def stop(self):
        """Finish the walk: the count becomes final, and the session is saved. Returns the session."""
        with self.lock:
            if self.mode != "walking":
                raise ValueError("Nothing is being counted: press START first.")
            self.process()
            if len(self.samples) < FS:
                self.mode = "idle"
                raise ValueError("That was under a second: nothing saved.")
            exact = self.counter.finish()
            steps = int(round(exact))
            active = self.counter.active_seconds()
            length, source = self.step_length(self.person, steps, active)
            length = round(length, 3) if length else length                     # the distance is steps x the step length shown, to the last digit
            n0, n1 = self.samples[0][0], self.samples[-1][0]
            expected = max(1, n1 - n0 + 1)
            dropped = max(0, expected - len(self.samples))
            stamp = self.t_wall.strftime("%Y%m%d-%H%M%S")
            sid, k = stamp, 2
            while any(s["id"] == sid for s in self.sessions):
                sid, k = "%s-%d" % (stamp, k), k + 1
            name = "%s__live__%s.csv" % (self.person.replace(" ", "_"), sid)
            with open(self.dir / name, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["n", "ax", "ay", "az", "gx", "gy", "gz"])
                for n, ax, ay, az, gx, gy, gz in self.samples:
                    w.writerow([n, "%.4f" % ax, "%.4f" % ay, "%.4f" % az, "%.3f" % gx, "%.3f" % gy, "%.3f" % gz])
            s = {"id": sid, "start": self.t_wall.strftime("%Y-%m-%d %H:%M:%S"), "person": self.person, "seconds": round(len(self.samples) / FS, 1), "steps": steps,
                 "steps_exact": round(exact, 2), "distance_est_m": round(steps * length, 2) if length else "", "step_length_m": round(length, 3) if length else "",
                 "step_length_source": source, "true_steps": "", "error": "", "true_distance_m": "", "file": name, "dropped": dropped, "lost_pct": round(100.0 * dropped / expected, 2)}
            self.sessions.append(s)
            self._save()
            self.mode, self.last = "done", dict(s)
            return dict(s)

    def reset(self):
        with self.lock:
            if self.mode == "walking":
                raise ValueError("Press STOP first.")
            self.mode, self.last = "idle", None

    def set_truth(self, session_id, true_steps, true_distance=None):
        """The steps you really counted (and the distance, if you measured it). Returns the updated session."""
        with self.lock:
            s = self._session(session_id)
            try:
                t = int(float(true_steps))
            except (TypeError, ValueError):
                raise ValueError("Type the number of steps you really counted (a whole number).")
            if float(true_steps) != t or t < 0 or t > MAX_STEPS:
                raise ValueError("The steps you counted must be a whole number from 0 to %d." % MAX_STEPS)
            dist = None
            if true_distance not in (None, ""):
                try:
                    dist = float(true_distance)
                except (TypeError, ValueError):
                    raise ValueError("The distance must be a number of metres, or empty.")
                if not 0.5 <= dist <= 500:
                    raise ValueError("The distance must be between 0.5 and 500 metres, or empty.")
                if t == 0:
                    raise ValueError("A walk with 0 steps has no distance: leave it empty.")
            s["true_steps"], s["error"] = t, s["steps"] - t
            if dist is not None:
                s["true_distance_m"] = dist
                lengths = self.calibration.setdefault(s["person"], [])
                lengths.append(dist / t)                               # this walk's own step length
                del lengths[:-10]
                self._write_calibration()
            self._save()
            if self.last and self.last["id"] == s["id"]:
                self.last = dict(s)
            return dict(s)

    def delete(self, session_id):
        with self.lock:
            s = self._session(session_id)
            try:
                (self.dir / s["file"]).unlink()
            except FileNotFoundError:
                pass
            self.sessions = [x for x in self.sessions if x["id"] != s["id"]]
            self._save()
            if self.last and self.last["id"] == s["id"]:
                self.mode, self.last = "idle", None

    def _session(self, session_id):
        for s in self.sessions:
            if s["id"] == session_id:
                return s
        raise ValueError("No such session.")

    # ---- distance ----
    def step_length(self, person, steps, active_s):
        """(metres per step, where it came from), or (None, why not)."""
        own = self.calibration.get(person) or self.calibration.get(clean_name(person))
        if own:
            return float(np.median(own)), "your own %d measured walk%s (median)" % (len(own), "" if len(own) == 1 else "s")
        if not self.distance:
            return None, "no step-length model"
        cad = float(steps) / max(float(active_s), 1e-6)
        models = self.distance.get("people", {})
        key = next((k for k in models if k.lower() == person.lower()), None)
        m = models[key] if key else self.distance
        length = min(max(m["a"] + m["b"] * cad, STEP_MIN), STEP_MAX)
        return length, ("the model fitted to %s's recorded walks" % key) if key else "the model for everyone (type a measured distance after a walk to calibrate)"

    # ---- saving and loading ----
    def _save(self):
        with open(self.dir / "live_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.sessions)
        with open(self.dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for s in self.sessions:
                w.writerow({"file": s["file"], "subject": s["person"], "kind": "live", "true_steps": s["true_steps"], "distance_m": s["true_distance_m"], "note": "live counter",
                            "start": s["start"], "seconds": s["seconds"], "samples": int(round(s["seconds"] * FS)), "dropped": s.get("dropped", 0), "lost_pct": s.get("lost_pct", 0)})

    def _load_log(self):
        path = self.dir / "live_log.csv"
        out = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if not r.get("file") or not (self.dir / r["file"]).exists():
                        continue
                    for k in ("steps", "true_steps", "error"):
                        r[k] = int(float(r[k])) if r.get(k) not in (None, "") else ""
                    for k in ("seconds", "steps_exact", "distance_est_m", "step_length_m", "true_distance_m"):
                        r[k] = float(r[k]) if r.get(k) not in (None, "") else ""
                    r["dropped"], r["lost_pct"] = 0, 0.0
                    r["step_length_source"] = ""
                    out.append(r)
        return out

    def _load_calibration(self):
        path = self.dir / "calibration.json"
        if path.exists():
            try:
                return {k: [float(x) for x in v] for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
            except (ValueError, TypeError):
                return {}
        return {}

    def _write_calibration(self):
        (self.dir / "calibration.json").write_text(json.dumps(self.calibration), encoding="utf-8")

    # ---- what the page shows ----
    def tally(self):
        """How right the counter has been in YOUR demos (sessions where you typed what you counted)."""
        done = [s for s in self.sessions if s["true_steps"] != ""]
        if not done:
            return {"n": 0}
        err = np.array([s["error"] for s in done], dtype=float)
        true = np.array([s["true_steps"] for s in done], dtype=float)
        pos = true > 0
        return {"n": len(done), "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)), "within1": float(np.mean(np.abs(err) <= 1)), "worst": float(np.max(np.abs(err))),
                "mape": float(100 * np.mean(np.abs(err[pos]) / true[pos])) if pos.any() else None}

    def snapshot(self):
        with self.lock:
            self.process()
            walking, done = self.mode == "walking", self.mode == "done"
            exact = self.counter.count if walking else (self.last["steps_exact"] if done and self.last else 0.0)
            steps = int(round(exact))
            active = self.counter.active_seconds() if walking else 0.0
            length, source = self.step_length(self.person, steps, active) if (walking and steps > 0) else (None, "")
            dist = steps * length if length else None
            if done and self.last:
                dist = self.last["distance_est_m"] if self.last["distance_est_m"] != "" else None
                length = self.last["step_length_m"] if self.last["step_length_m"] != "" else None
                source = self.last.get("step_length_source", "")
            rate = self.imu_rate()
            d = np.concatenate([self.counter.density, self.counter.tail]) if walking else np.zeros(0)
            d = d[-15 * int(FS):]
            k = 10
            dens = [round(float(d[i:i + k].mean() * FS), 2) for i in range(0, len(d) - k + 1, k)] if len(d) >= k else []
            tail = self.samples[-15 * int(FS):] if walking else []
            gyro = [round(float(np.mean([math.sqrt(r[4] ** 2 + r[5] ** 2 + r[6] ** 2) for r in tail[i:i + k]])), 1) for i in range(0, len(tail) - k + 1, k)] if len(tail) >= k else []
            return {"mode": self.mode, "person": self.person, "steps": steps, "steps_exact": round(exact, 2), "elapsed": round(len(self.samples) / FS, 1) if (walking or done) else 0,
                    "pace": round(self.counter.steps_per_minute(), 0) if walking else 0, "distance_m": round(dist, 2) if dist is not None else None,
                    "step_length_m": round(length, 3) if length else None, "step_length_source": source, "imu_rate": rate, "imu_ok": rate >= MIN_IMU_RATE,
                    "plot": {"density": dens, "gyro": gyro}, "last": dict(self.last) if self.last else None, "tally": self.tally(),
                    "sessions": [dict(s) for s in reversed(self.sessions[-8:])], "people": sorted(self.distance.get("people", {}).keys()), "evaluation": self.evaluation}


# ---- the glue: the page, the sketch --------------------------------------------------------------------------------
net = NumpyStepNet.load(MODEL_DIR / "step_net.json")
distance = json.loads((MODEL_DIR / "distance.json").read_text(encoding="utf-8")) if (MODEL_DIR / "distance.json").exists() else {}
evaluation = json.loads((MODEL_DIR / "evaluation.json").read_text(encoding="utf-8")) if (MODEL_DIR / "evaluation.json").exists() else {}
live = Live(net, distance, evaluation, DATA_DIR)
print("network with %d weights, %d past sessions, data in %s" % (net.n_parameters(), len(live.sessions), DATA_DIR), flush=True)


def on_imu(n, ax, ay, az, gx, gy, gz):
    """Called by the sketch for every reading, on the Bridge's own thread: keep it light (the counting happens in the loop)."""
    live.add_sample(n, ax, ay, az, gx, gy, gz)


Bridge.provide("imu", on_imu)
ui = WebUI()
led = {"sent": None, "tick": 0}


def notice(level, text):
    ui.send_message("notice", {"level": level, "text": text})


def on_cmd(sid, data):
    try:
        action = (data or {}).get("action")
        if action == "start":
            live.start(data.get("person"))
            notice("info", "Counting. Walk, then press STOP.")
        elif action == "stop":
            s = live.stop()
            notice("ok", "Counted %d steps%s. Type how many you really counted, so the app can tell how right it was." % (s["steps"], "" if s["distance_est_m"] == "" else " (about %.1f m)" % s["distance_est_m"]))
        elif action == "truth":
            s = live.set_truth(data.get("id"), data.get("steps"), data.get("distance"))
            notice("ok", "Saved: the counter said %d, you counted %d (%s)." % (s["steps"], s["true_steps"], "exactly right" if s["error"] == 0 else "%+d" % s["error"]))
        elif action == "delete":
            live.delete(data.get("id"))
            notice("info", "Session deleted.")
        elif action == "reset":
            live.reset()
        elif action != "refresh":
            notice("warn", "Unknown command.")
            return
        ui.send_message("state", live.snapshot())
    except ValueError as e:
        notice("warn", str(e))
    except Exception as e:                       # never let a bad command kill the app
        notice("warn", "Something went wrong: " + str(e)[:120])


ui.on_message("cmd", on_cmd)
ui.on_connect(lambda sid: ui.send_message("state", live.snapshot()))


def loop():
    time.sleep(0.2)
    snap = live.snapshot()
    ui.send_message("state", snap)
    want = (4, snap["steps"]) if snap["mode"] in ("walking", "done") else (0, 0)
    led["tick"] += 1
    if want != led["sent"]:
        try:
            Bridge.call("led", want[0], want[1], timeout=2)
            led["sent"] = want
        except Exception as e:
            if led["tick"] % 100 == 0:
                print("(LED matrix not answering: " + str(e)[:60] + ")", flush=True)


App.run(user_loop=loop)
