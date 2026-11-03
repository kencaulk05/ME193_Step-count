# The first run, before the fix

This is what the network did on your 55 trials the first time, when its training could collapse to "count nothing". It is kept because it is a real result and a useful one.

| | Network | Classical baseline |
|---|---|---|
| Average error in cross-validation (steps) | 2.20 (26.5% of the count) | 0.76 (7.2%) |
| Trained without Kendrick, tested on Kendrick (steps off) | 8.50 | 0.71 |
| Trained without marie, tested on marie (steps off) | 9.33 | 3.04 |

The errors were not spread evenly. **8 walking trials were predicted as 0.0 steps** (a count under half a step; errors of -20 to -9), because the networks that predicted them had collapsed to counting nothing (their predictions were between 0.00001 and 0.0005 steps). In a separate test on random splits of the same trials, collapsed networks ended with a training loss near 1.1 instead of 0.06 and counted 0% of the true steps in their own training trials; the first run's own training losses were not saved. The other trials were typically within a step or two. The cause was the starting point of training (see README.md, section 2 of "Things that went wrong"). After the fix, no network collapsed, and the average error fell from 2.20 to 0.67 steps.
