# Tests

```
python tests/test_steps.py        (or: python run_all_tests.py)
```

What they check, in order: loading the recorder's zip (skipped trials, each with its reason; bad files refused); the 8 numbers the network reads; that the NumPy copy of the convolution equals a slow loop-by-loop version; the classical baseline on the REAL standing and air-biking recordings (absolute thresholds count nothing on a motionless sensor, and do count air-biking, which is what the network must learn not to do); the distance model; the cross-validation folds and the error statistics; the live counter (its final count equals the offline count exactly, however the readings arrive, and a network with a known answer is counted exactly); then, with PyTorch installed, the network, the loss arithmetic, the augmentations, training, determinism, the NumPy export matching PyTorch, and the whole `train_steps.py` command on synthetic trials.

The synthetic trials have a known right answer by construction. They check that the machinery works. They are not evidence about real accuracy: that comes only from trials recorded on a real ankle.
