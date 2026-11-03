# Step counter: how good is it?

Every number below is out-of-fold: each trial was predicted by a network that never saw it. The truth is the count you typed, and that count can itself be off by a step or two.

## Data

55 trials, 2 person(s), 10.5 minutes, 490 steps counted.

| Kind | Trials | Steps | Network average error (steps) |
|---|---|---|---|
| foot_tap | 8 | 0 | 0.00 |
| jog | 10 | 95 | 1.02 |
| other | 7 | 140 | 0.82 |
| stand | 6 | 0 | 0.00 |
| walk_fast | 8 | 78 | 0.60 |
| walk_normal | 8 | 84 | 1.02 |
| walk_slow | 8 | 93 | 0.96 |

## Network against the classical baseline

| | Network | Baseline (gyro peaks above 100 degrees/s at least 0.6 s apart) |
|---|---|---|
| Average error, steps (95% interval) | **0.67** (0.47 to 0.89) | 0.76 (0.47 to 1.13) |
| Average error, % of the count (95% interval) | **8.0%** (5.8 to 10.5) | 7.2% (5.4 to 9.0) |
| Average signed error (bias), steps | -0.13 | +0.00 |
| Spread of the error (SD), steps | 1.02 | 1.45 |
| Average error on the 41 trials WITH steps only, steps | 0.89 | 0.78 |
| Trials within 1 step | 75% | 91% |
| Trials within 2 steps | 89% | 95% |
| Worst trial, steps off | 2.8 | 8.0 |
| Steps counted in a trial with NO steps (average, worst) | 0.0, 0.0 | 0.7, 8.0 |

## A new person

Trained on everyone else, tested on this person.

| Tested on | Trials | Network average error (steps) | Baseline average error (steps) |
|---|---|---|---|
| Kendrick | 28 | 1.19 | 0.71 |
| marie | 27 | 1.16 | 3.04 |

## Distance walked

34 walks of a measured distance, each predicted by a model that never saw it. Distance is the number of steps times the length of a step (step length = 0.420 + 0.044 x cadence).

| | Average error |
|---|---|
| Using your true step counts (the step-length model alone) | 8.8% |
| End to end (the network's count x step length) | **11.5%** (bias +0.5%, SD 15.7%, worst 51.0%) |
| End to end, step length calibrated on Kendrick's own walks only (18 walks; step length = 0.483 + 0.024 x cadence) | **7.2%** (bias +3.6%, SD 9.0%, worst 24.9%) |
| End to end, step length calibrated on marie's own walks only (16 walks; step length = 0.275 + 0.103 x cadence) | **15.3%** (bias -4.3%, SD 19.8%, worst 41.2%) |

People have different step lengths, so one model for everyone does worse than a model calibrated on the wearer's own measured walks.

## Limits to say out loud

- The truth is your own count, good to about a step or two, so errors smaller than that are not meaningful.
- Training can get stuck counting nothing. 0 of 5 cross-validation networks needed a restart with another seed (a stuck run is spotted from the training trials alone: the network counts almost none of their steps. The test trials are never used for that.)
- The training fix (starting the step density very low, so that training does not collapse to counting nothing) was chosen by looking at random splits of these same trials, so the numbers above are slightly optimistic. The first run, before the fix, is described in README.md.
- About 55 trials and 2 person(s): the intervals above show how much the numbers would move with different trials, and they are wide when there are few.
- The sensor must sit the same way on the same ankle, on the same kind of walking you recorded (corridor, normal shoes). Stairs, slopes, carrying things and other shoes were not tested.
- Step length changes with speed, turns, and the person; the distance error grows with distance and was measured on short straight courses.

## Files

`scatter_network.png`, `errors_network.png`, `network_vs_baseline.png`, `example_typical.png`, `example_worst.png`, `training_loss.png`, `distance.png`, `results.json` (every trial's prediction).
