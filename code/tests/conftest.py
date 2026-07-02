"""
pytest configuration: adds code/ to sys.path so all test modules can import
project packages (agents, environment, data, evaluation, ...) without an
externally set PYTHONPATH.
"""

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
