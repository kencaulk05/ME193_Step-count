# Tests (no board needed)

```
python run_all.py          from this folder: runs every group whose tools are installed, and says which it skipped
```

Or one at a time:

```
g++ -std=c++17 -I stubs -o test_sketch test_sketch.cpp && ./test_sketch    the sketch against a simulated MPU-6050 and LED matrix (the IMU stream, and the number display)
python test_live_app.py                                                    python/main.py, with the real network; writes state_samples.json
npm install jsdom && node test_page.js                                     the web page in a simulated browser, rendering state_samples.json
```

If `all_trials.zip` is found (set `STEPS_TRIALS=all_trials.zip`), `test_live_app.py` also streams real trials through the app, one reading at a time, and checks that the final count equals the offline count of the same trial.
