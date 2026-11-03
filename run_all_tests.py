"""Run the step counter's tests. Usage: python run_all_tests.py   (PyTorch tests are skipped, and say so, if PyTorch is not installed)"""
import subprocess
import sys
from pathlib import Path

p = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "tests" / "test_steps.py")])
sys.exit(p.returncode)
