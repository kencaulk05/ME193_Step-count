"""Run the step counter's tests. Usage: python run_all_tests.py   (PyTorch tests are skipped, and say so, if PyTorch is not installed)"""
import os
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
codes = [subprocess.run([sys.executable, str(here / "tests" / "test_steps.py")]).returncode]
app_tests = here / "apps" / "gait-steps-live" / "tests" / "run_all.py"
if app_tests.exists():
    print("\n=== the live app ===")
    codes.append(subprocess.run([sys.executable, str(app_tests)], cwd=app_tests.parent, env=dict(os.environ, STEPS_TRIALS=os.environ.get("STEPS_TRIALS", ""))).returncode)
sys.exit(1 if any(codes) else 0)
