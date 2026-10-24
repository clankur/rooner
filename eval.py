"""Evaluate Tinker policies against the same verified live runner used for RLVR.

Example: python eval.py --episodes 10 --checkpoint tinker://.../sampler_weights/...
Legacy local .pt checkpoints are not Tinker checkpoints.
"""
import sys

from rsenv.rlvr.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
