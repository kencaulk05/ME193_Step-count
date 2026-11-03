"""Trials: loading what the step recorder saved, and turning a recording into the numbers the network reads.

A TRIAL is one recording of the ankle IMU (100 readings a second: ax, ay, az in g, gx, gy, gz in degrees/second) together with the truth
you counted: how many steps you took (both feet), and how far you walked if you measured it.

    load_trials("all_trials.zip")        the zip from the recorder's "Prepare zip of everything" button, or a folder holding manifest.csv
    load_legacy_negatives("data/")       the older gait recordings: standing still and air-biking have NO steps, so they are free negative examples
"""
import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = 100.0                                   # readings per second
ACCEL_SCALE = 4.0                            # g
GYRO_SCALE = 500.0                           # degrees per second
N_FEATURES = 8                               # ax, ay, az, gx, gy, gz, |a|-1, |g|
MAX_LOST_PCT = 2.0
TRIM_S = 0.0                                 # trials already start and end quietly, so nothing is trimmed


@dataclass
class Trial:
    name: str
    person: str
    kind: str
    steps: int                               # the truth: steps taken, both feet
    distance_m: float                        # metres walked, or nan when not measured
    x: np.ndarray                            # (T, 6) float32
    source: str = "recorder"                 # "recorder", "legacy" or "synthetic"

    @property
    def seconds(self):
        return len(self.x) / FS


def _read_csv_text(text, name):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0][:7] != ["n", "ax", "ay", "az", "gx", "gy", "gz"]:
        raise ValueError("%s: not a recorder file (expected the header n,ax,ay,az,gx,gy,gz)" % name)
    a = np.array([[float(v) for v in r[1:7]] for r in rows[1:] if len(r) >= 7], dtype=np.float32)
    if len(a) < 100:
        raise ValueError("%s: fewer than 100 readings" % name)
    return a


def _files_of(source):
    """(manifest text, {file name: text}) from a zip or a folder."""
    source = Path(source)
    if source.is_dir():
        files = {p.name: p.read_text(encoding="utf-8") for p in source.glob("*.csv")}
    else:
        with zipfile.ZipFile(source) as z:
            files = {Path(n).name: z.read(n).decode("utf-8") for n in z.namelist() if n.lower().endswith(".csv")}
    if "manifest.csv" not in files:
        raise ValueError("no manifest.csv in %s: use the zip from the recorder's 'Prepare zip of everything' button" % source)
    return files.pop("manifest.csv"), files


def load_trials(source, include_unlabelled=False):
    """All good trials with a step count. Skips (and reports) trials that lost too many readings or still need their step count."""
    manifest, files = _files_of(source)
    trials, skipped = [], []
    for r in csv.DictReader(io.StringIO(manifest)):
        name = r.get("file", "")
        if name not in files:
            skipped.append((name, "file missing from the zip")); continue
        if float(r.get("lost_pct") or 0) > MAX_LOST_PCT:
            skipped.append((name, "more than %g%% of the readings were lost" % MAX_LOST_PCT)); continue
        if (r.get("true_steps") or "") == "" and not include_unlabelled:
            skipped.append((name, "no step count yet: press Edit on it in the recorder")); continue
        steps = int(float(r["true_steps"])) if (r.get("true_steps") or "") != "" else -1
        dist = float(r["distance_m"]) if (r.get("distance_m") or "") != "" else float("nan")
        trials.append(Trial(name=name, person=r.get("subject", "?"), kind=r.get("kind", "?"), steps=steps, distance_m=dist, x=_read_csv_text(files[name], name)))
    return trials, skipped


def load_legacy_negatives(folder, chunk_s=30.0):
    """The earlier gait recordings: 'still' and 'bike' (air-biking on the back) have no steps, so they teach the network what NOT to count.
    Each is cut into chunks, and the first and last second (starting and stopping) are dropped."""
    out = []
    for p in sorted(Path(folder).glob("*__*__*.csv")):
        person, label = p.name.split("__")[:2]
        if label not in ("still", "bike"):
            continue
        a = _read_csv_text(p.read_text(encoding="utf-8"), p.name)[int(FS):-int(FS)]
        n = int(chunk_s * FS)
        for i in range(0, len(a) - n // 2, n):
            seg = a[i:i + n]
            if len(seg) >= n // 2:
                out.append(Trial(name="%s#%d" % (p.name, i // n), person=person, kind="stand" if label == "still" else "bike", steps=0, distance_m=float("nan"), x=seg, source="legacy"))
    return out


def features(x):
    """(T, 6) raw readings -> (T, 8) numbers on a similar scale: ax, ay, az in units of 4 g, gx, gy, gz in units of 500 degrees/s,
    then the total acceleration minus gravity and the total rotation."""
    x = np.asarray(x, dtype=np.float32)
    a, g = x[:, :3], x[:, 3:]
    am = np.linalg.norm(a, axis=1, keepdims=True)
    gm = np.linalg.norm(g, axis=1, keepdims=True)
    return np.concatenate([a / ACCEL_SCALE, g / GYRO_SCALE, (am - 1.0) / 2.0, gm / GYRO_SCALE], axis=1).astype(np.float32)
