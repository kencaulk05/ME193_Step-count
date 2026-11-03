# How many steps? The step counter

A small neural network, written in PyTorch, that counts steps (both feet) from the ankle IMU, and a distance estimate built on top of it.

## The idea in four lines

1. The IMU sends 8 numbers every 10 ms (acceleration and rotation on three axes, plus their totals).
2. Five layers of 1-D convolution, each looking at neighbouring moments of that signal, turn them into **one number per moment: how many steps are happening right now** (a step density, never negative).
3. **Add the density up over a trial and you have the step count.**
4. Training only needs the number you counted for each trial. There is no label saying *where* each step is. The loss is how far the sum is from your count (on a log scale), so the network has to discover what a step looks like, and that standing, a tapping foot and air-biking are not steps.

```
signal --> conv --> ReLU --> conv --> ReLU --> conv --> ReLU --> conv --> ReLU --> conv --> ReLU --> 1x1 conv --> softplus --> density
 (8 x time)                                                                                                     (1 x time)
loss = | ln(1 + sum(density)) - ln(1 + steps you counted) |      backpropagation + Adam update every weight
```

About 6000 weights, a receptive field of 125 readings (1.25 s). Training is on whole trials, with augmentations that cannot change the step count: the sensor tilted a few degrees, a different strength, the same steps faster or slower, more noise.

## Use it

```
pip install -r requirements.txt                        # on Windows and Mac the default torch should be the CPU build; that is all this needs
python train_steps.py --trials all_trials.zip          # the zip from the step recorder; about 10 minutes (60 epochs)
python train_steps.py --trials all_trials.zip --quick  # a rougher run in about 2 minutes, to check everything works
python run_all_tests.py
```

`train_steps.py` loads the trials, tests the network against a classical peak-counting baseline with **cross-validation** (every trial is predicted by a network that never saw it), repeats the test on a **new person** when there are two, estimates **distance** (steps times a step length fitted to the measured walks, tested the same honest way), trains the final network, and writes:

| Output | What it is |
|---|---|
| `models/step_net.json` | the trained network as plain numbers; `steps_numpy.py` runs it with no PyTorch (the board uses this) |
| `models/step_net.pt` | the same weights for PyTorch |
| `models/distance.json` | the step-length model |
| `results/summary.md` | every number with its uncertainty, and the limits |
| `results/*.png` | predicted against counted steps, the errors, network against baseline, example trials, training loss, distance |
| `results/results.json` | every trial's out-of-fold prediction |

## Files

```
steps_data.py       load the recorder's zip; the 8 numbers the network reads
steps_model.py      the PyTorch network, the augmentations, the loss, the training loop
steps_numpy.py      the same network in NumPy
steps_live.py       counting while walking: exact once 0.62 s of signal after a moment has arrived
steps_baseline.py   the classical counter (peaks above an absolute threshold)
steps_distance.py   step length from cadence, with leave-one-out error
steps_eval.py       cross-validation, error statistics, bootstrap intervals
steps_plots.py      the figures
synth.py            made-up trials with a known right answer, for the tests only
train_steps.py      the command above
tests/              what the tests check is in tests/README.md
```

## Results on the real trials

55 trials from two people (10.5 minutes, 490 steps counted), every number out-of-fold: each trial predicted by a network that never saw it. The truth is the count typed in the recorder, good to about a step or two.

| | Network | Classical baseline |
|---|---|---|
| Average error, steps (95% interval) | **0.67** (0.47 to 0.89) | 0.76 (0.47 to 1.13) |
| Average error on the 41 trials with steps only | 0.89 | 0.78 |
| Steps counted on the 14 trials with no steps (standing, foot tapping) | 0.0 on average, 0 at worst | 0.7 on average, 8 at worst |
| Trials within 1 step | 75% | 91% |
| Worst trial, steps off | 2.8 | 8.0 |
| Trained without Kendrick, tested on Kendrick (steps off) | 1.19 | 0.71 |
| Trained without marie, tested on marie (steps off) | 1.16 | 3.04 |

Read this honestly: **on trials with steps, the classical counter is about as good as the network** (slightly better on average, and better at being within 1 step). The network's overall edge comes from the no-step trials: it counted nothing on all of them, while the classical counter counted false steps on two foot-tapping trials. It also carries over to a new person more evenly (about 1.2 steps either way, against 0.7 and 3.0). It under-counts jogging by about 2 steps in some trials.

**Distance.** Steps times a step length fitted to the measured walks: about 11.5% off end to end with one model for both people, 7.2% off when calibrated on one person's own walks (Kendrick), and 15.3% for the other (marie), whose step counts did not track the distance as consistently (13 steps for both the 4.8 m and the 5.8 m line). The walks were only 4.8 to 5.8 m long, so one step out is about a 10% error on its own.

The training fix described below was chosen by looking at random splits of these same trials, so these numbers are slightly optimistic. `results/FIRST_RUN_BEFORE_THE_FIX.md` records the first run, before the fix, when training collapsed.

## Things that went wrong, and what fixed them

**1. The first loss function.** The first loss was the relative error, `|counted - true| / (true + 5)`. On made-up trials with a known answer the network **collapsed**: it learned to count nothing at all, so every walk was predicted as 0.0 steps. On a trial with no steps, one early false step cost 12 times as much as missing steps on a walk, so the cheapest thing was to output zero everywhere, and a softplus that far below zero has almost no gradient to climb back out. The log loss above keeps a false step expensive but bounded, and the same test then gave 6% error, 0.84 steps on average and no false steps on the 0-step trials. A higher learning rate (6e-3) was unstable, so it stays at 3e-3. The test that would catch this again is in `tests/test_steps.py` (section 11).

**2. Training that still got stuck, on the real trials.** With the log loss, training still collapsed to "count nothing" in 2 of 5 runs on the real recordings (the final training loss was 1.1 instead of 0.06, and the network counted 0% of the true steps). The cause was the starting point: the density began at 0.018 steps per reading, which is 36 false steps in a 20-second trial with no steps, so the early updates turned everything down, and a softplus far below zero has almost no gradient to come back up. Starting at 0.0009 per reading (`INIT_BIAS = -7`) removed it: 0 of 5 collapses, and the average held-out error on those splits fell from 3.9 to 0.6 steps. As a safety net, a run whose network counts less than 30% of the true steps on its own training walks is discarded and restarted with another seed, and the summary reports how often that happened (the training loss is not used for this: a healthy run can still be improving slowly). The test is section 11b.

## Limits

- The truth is the count you typed, good to about a step or two.
- A few dozen trials from one or two people: read the intervals in `summary.md`, not just the averages.
- The sensor must sit the same way on the same ankle, on the kind of walking that was recorded. Stairs, slopes, carrying things and other shoes were not tested.
- Step length depends on speed and on the person, and distance error grows with distance.
