"""Set before pytest imports torch: cuBLAS needs this for deterministic
reductions, and train/run.py asserts bitwise reproducibility."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
