"""
Standalone diagnostic script to test if joblib/pickle can read the model files.
Run this directly with: python debug_models.py
It does NOT touch Flask, SocketIO, or any of the app code - pure isolation test.
"""

import os
import sys
import hashlib
import pickletools

MODEL_DIR = r"C:\Users\Nasser\Desktop\DDos-Windows\Models"

FILES = [
    "random_forest.pkl",
    "xgboost.pkl",
    "isolation_forest.pkl",
    "scaler.pkl",
    "feature_names.pkl",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def print_env():
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print("Python:", sys.version)
    try:
        import sklearn
        print("scikit-learn:", sklearn.__version__)
    except Exception as e:
        print("scikit-learn: FAILED TO IMPORT ->", e)
    try:
        import joblib
        print("joblib:", joblib.__version__)
    except Exception as e:
        print("joblib: FAILED TO IMPORT ->", e)
    try:
        import xgboost
        print("xgboost:", xgboost.__version__)
    except Exception as e:
        print("xgboost: FAILED TO IMPORT ->", e)
    try:
        import numpy
        print("numpy:", numpy.__version__)
    except Exception as e:
        print("numpy: FAILED TO IMPORT ->", e)
    try:
        import scipy
        print("scipy:", scipy.__version__)
    except Exception as e:
        print("scipy: FAILED TO IMPORT ->", e)
    print()


def check_file_basics(path):
    """Check existence, size, first bytes (pickle protocol magic), lock status."""
    print(f"--- {os.path.basename(path)} ---")
    if not os.path.exists(path):
        print("  STATUS: FILE NOT FOUND")
        return False

    size = os.path.getsize(path)
    print(f"  Size: {size} bytes")

    # Check if it's a OneDrive/cloud placeholder (reparse point) even off Desktop
    try:
        import stat
        st = os.stat(path)
        attrs = getattr(st, "st_file_attributes", None)
        if attrs is not None:
            FILE_ATTRIBUTE_REPARSE_POINT = 0x400
            FILE_ATTRIBUTE_OFFLINE = 0x1000
            if attrs & FILE_ATTRIBUTE_REPARSE_POINT:
                print("  WARNING: File is a REPARSE POINT (cloud placeholder, e.g. OneDrive)")
            if attrs & FILE_ATTRIBUTE_OFFLINE:
                print("  WARNING: File is marked OFFLINE (not fully downloaded)")
    except Exception as e:
        print(f"  (could not check file attributes: {e})")

    # Read raw bytes and inspect pickle protocol header
    try:
        with open(path, "rb") as f:
            head = f.read(16)
        print(f"  First 16 bytes (hex): {head.hex()}")
        # Standard pickle protocol 2 starts with b'\x80\x02'
        if head[:1] == b"\x80":
            print(f"  Pickle protocol byte detected: {head[1]}")
        else:
            print("  WARNING: Does not start with standard pickle opcode (0x80). "
                  "File may be corrupted, truncated, or not a real pickle.")
    except Exception as e:
        print(f"  ERROR reading raw bytes: {e}")
        return False

    print(f"  SHA256: {sha256(path)}")
    return True


def try_joblib_load(path):
    import joblib
    try:
        obj = joblib.load(path)
        print(f"  joblib.load(): SUCCESS -> {type(obj)}")
        return True
    except Exception as e:
        print(f"  joblib.load(): FAILED -> {type(e).__name__}: {e}")
        return False


def try_raw_pickle_load(path):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  raw pickle.load(): SUCCESS -> {type(obj)}")
        return True
    except Exception as e:
        print(f"  raw pickle.load(): FAILED -> {type(e).__name__}: {e}")
        return False


def try_pickletools_disassemble(path):
    """If both loaders fail, try to disassemble the opcode stream to see where it breaks."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        import io
        pickletools.dis(io.BytesIO(data))
        print("  pickletools.dis(): completed without error (stream is structurally valid)")
    except Exception as e:
        print(f"  pickletools.dis(): FAILED -> {type(e).__name__}: {e}")
        print("  This tells us roughly how many bytes were successfully parsed before failure.")


if __name__ == "__main__":
    import pickle

    print_env()

    print("=" * 60)
    print("FILE CHECKS")
    print("=" * 60)
    results = {}
    for fname in FILES:
        path = os.path.join(MODEL_DIR, fname)
        ok = check_file_basics(path)
        print()
        results[fname] = ok

    print("=" * 60)
    print("LOAD TESTS")
    print("=" * 60)
    for fname in FILES:
        path = os.path.join(MODEL_DIR, fname)
        if not results.get(fname):
            print(f"--- {fname}: SKIPPED (failed basic checks) ---\n")
            continue
        print(f"--- {fname} ---")
        ok = try_joblib_load(path)
        if not ok:
            try_raw_pickle_load(path)
            try_pickletools_disassemble(path)
        print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)
