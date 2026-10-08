#!/usr/bin/env python3
"""Run the same read-only acceptance on ARM under a separate concurrency key."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("pr515_acceptance.py")), run_name="__main__")
