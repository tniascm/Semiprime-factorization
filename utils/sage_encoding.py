"""
Shared Sage/NumPy type encoding utilities for JSON serialization.

Consolidates SageEncoder and _py() which were duplicated across 10+ experiment files.
"""
import json
import numpy as np


class SageEncoder(json.JSONEncoder):
    """JSON encoder that handles Sage Integer, RealDoubleElement, numpy types, etc."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return int(obj)
        except (TypeError, ValueError):
            pass
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)


def _py(v):
    """Convert Sage/NumPy types to native Python types for JSON serialization."""
    if isinstance(v, (bool, type(None), str)):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, np.float64)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    # Sage RealDoubleElement, Integer, Rational, etc.
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def _py_dict(d):
    """Recursively convert all values in a dict for JSON."""
    return {k: _py(v) for k, v in d.items()}


def safe_json_dump(data, filepath, indent=2):
    """Write JSON with SageEncoder, printing to stdout as backup on failure."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, cls=SageEncoder, indent=int(indent))
        print(f"Results saved to {filepath}", flush=True)
    except (IOError, OSError) as e:
        print(f"WARNING: Failed to write {filepath}: {e}", flush=True)
        print("Dumping results to stdout as backup:", flush=True)
        print(json.dumps(data, cls=SageEncoder, indent=int(indent)), flush=True)
