"""Run every test of this app that the computer can run. Usage: python run_all.py   (from the tests folder)"""
import os, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXE = ".exe" if os.name == "nt" else ""
results = []


def run(label, cmd):
    print(f"\n=== {label} ===", flush=True)
    p = subprocess.run(cmd, cwd=HERE)
    results.append((label, "passed" if p.returncode == 0 else "FAILED"))


def skip(label, why):
    print(f"\n=== {label} ===\n  skipped: {why}")
    results.append((label, "skipped (" + why + ")"))


if shutil.which("g++"):
    built = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", "stubs", "-o", "test_sketch" + EXE, "test_sketch.cpp"], cwd=HERE)
    if built.returncode == 0:
        run("sketch against a simulated MPU-6050 and LED matrix", [str(HERE / ("test_sketch" + EXE))])
    else:
        results.append(("sketch", "FAILED (did not compile)"))
else:
    skip("sketch", "g++ is not installed")
run("python/main.py (the live counter's logic)", [sys.executable, "test_live_app.py"])
node = shutil.which("node")
if node and subprocess.run([node, "-e", "require('jsdom')"], cwd=HERE, capture_output=True).returncode == 0:
    run("web page", [node, "test_page.js"])
else:
    skip("web page", "node or the jsdom package is missing (npm install jsdom)")

print("\n=== summary ===")
for label, outcome in results:
    print(f"  {label:<52} {outcome}")
bad = [r for r in results if r[1].startswith("FAILED")]
print("\n" + ("SOMETHING FAILED" if bad else f"ALL PASSED: {sum(r[1] == 'passed' for r in results)} groups passed, {sum(r[1].startswith('skipped') for r in results)} skipped"))
sys.exit(1 if bad else 0)
