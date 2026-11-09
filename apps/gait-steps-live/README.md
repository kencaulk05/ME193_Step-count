# How many steps? The live counter

Walk with the ankle sensor on, and this app counts your steps as you go. The count is on the web page (port 7000) and as a number on the board's LED matrix. After each walk you tell it how many steps you really counted, and it keeps a running tally of how right it was.

## Wiring and setup

The same as the step recorder: MPU-6050 on the header SDA and SCL pins, powered from **3V3** (not 5V), on the same ankle and the same way round as when the trials were recorded. Copy this folder into Arduino App Lab as an app and press Run. Then open `http://<the board's address>:7000` on a phone or laptop on the same network.

## One walk

1. Type who is wearing it, then press **START** and walk. The count goes up as you walk. The last half second of it is an estimate that can change a little.
2. Press **STOP**. The count becomes final: exactly what the network says for the whole walk. The LED matrix shows it as a number (two digits; 99 or more shows 99 with a line under it).
3. Type **the steps you really counted** (count out loud), and the distance if you measured it. The page says how far off the counter was, and the tally updates.
4. Press **Next walk**.

## What it shows

| | |
|---|---|
| Count | steps so far, both feet, rounded to a whole step |
| Steps a minute | the pace over the last 10 seconds |
| Metres | steps times a step length |
| The plot | what the network sees: its step density (green) over the rotation of the ankle (orange) |
| Tally | for the walks where you typed your count: how many, the average error, how many were within one step, the worst |
| How good is it | the numbers from the 55 test trials, copied from the evaluation that came with the network |

## How the distance is worked out

Distance is steps times a step length. The step length comes from the first of these that exists:

1. **Your own calibration**: the median of the walks where you typed a measured distance. Walk a taped line, count your steps, type the distance and the count. Do it two or three times; the median is robust to one bad walk. The last 10 are kept.
2. **The model fitted to that person's recorded trials** (a line: step length = a + b x cadence), if the name matches someone in the training data.
3. **The model for everyone.**

The page says which one it used. People have different step lengths, so a calibration on your own walks beats the model for everyone (in the tests: 7% against 11% for one person, and 15% for the other, whose step counts did not follow the distance as consistently).

## Names

The step-length models are called A and B (the order the two people recorded), never by a name, and the page suggests those letters. To use A's model when you type your own name, put a file `assets/data/person_aliases.json` next to the sessions, like `{"yourname": "A"}`. That folder is git-ignored (the sessions carry whatever name you typed), and so is the alias file: keep them out of git.

## What is saved

In `assets/data/`, next to the page:

| File | |
|---|---|
| `<name>__live__<date>-<time>.csv` | every reading of the walk: `n, ax, ay, az, gx, gy, gz`, the step recorder's format |
| `manifest.csv` | the recorder's manifest, with kind `live` and the count you typed; `train_steps.py` can read this folder directly, so your demos can become training data |
| `live_log.csv` | one row per walk: what it counted, what you counted, the error, the distance estimate |
| `calibration.json` | each person's measured step lengths |

Everything survives a restart. **Delete this walk** removes a bad session (a false start) from all of these.

## Honest limits

- **Whole-walk answer vs live answer.** STOP gives exactly the offline answer. While you walk, the last 0.62 seconds are estimated, because the network looks 0.62 seconds ahead of each moment.
- **It counts what it was trained on:** ankle sensor, same orientation, indoor walking and jogging, a corridor or flat. Stairs, slopes, carrying things and other shoes were not tested.
- **Short walks.** The test trials were 9 to 20 steps. One step is about 10% of a 10-step walk.
- **If readings are lost** (the page shows when the sensor stops), the count pauses, and the lost fraction is written to the manifest.
- **The truth is you.** The tally is only as good as your counting.

## Files

```
app.yaml, sketch/        the IMU stream at 100 readings a second, plus the LED matrix (number mode)
python/main.py           the app: readings in, counting, sessions, tally, calibration, the page and the LED
python/steps_*.py        the network in NumPy and the live counter (copied from the project by sync_live_app.py)
python/model/            the trained network, the step-length models, the evaluation numbers
assets/                  the web page
tests/                   see tests/README.md
```

After retraining, run `python sync_live_app.py` in the project folder to copy the new network into this app.
