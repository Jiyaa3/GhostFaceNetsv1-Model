"""
Group_13_1.py - Entry point for the GhostFaceNet real-time facial recognition system.
Run:  py Group_13_1.py
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from app import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
