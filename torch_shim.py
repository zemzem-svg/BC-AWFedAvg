"""
Minimal torch compatibility shim for numpy-only environments.

Only covers the subset of torch used by this project's non-training modules
(secure_aggregation, robustness_module, efficient_dp).  This is NOT a full
torch replacement — it exists solely to allow validation tests to run in
environments where PyTorch is not installed.

Usage:
    try:
        import torch
    except ImportError:
        import torch_shim as torch
"""

from __future__ import annotations
import numpy as np
from collections import OrderedDict


# ─────────────────────────────────────────────────────────────────────────────
# dtype constants
# ─────────────────────────────────────────────────────────────────────────────
float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64


# ─────────────────────────────────────────────────────────────────────────────
# Tensor — thin wrapper around ndarray
# ─────────────────────────────────────────────────────────────────────────────

class Tensor:
    """Minimal ndarray wrapper that supports basic torch-like operations."""

    def __init__(self, data, dtype=None):
        if isinstance(data, Tensor):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data if dtype is None else data.astype(dtype)
        else:
            self._data = np.array(data, dtype=dtype or np.float32)

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def numel(self):
        return self._data.size

    # ── Conversions ───────────────────────────────────────────────────────
    def numpy(self):
        return self._data.copy()

    def clone(self):
        return Tensor(self._data.copy())

    def float(self):
        return Tensor(self._data.astype(np.float32))

    def flatten(self):
        return Tensor(self._data.ravel())

    def to(self, dtype):
        return Tensor(self._data.astype(dtype))

    def item(self):
        return self._data.item()

    def __float__(self):
        return float(self._data)

    def __gt__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return bool(np.all(self._data > o)) if self._data.ndim == 0 else Tensor(self._data > o)

    def __lt__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return bool(np.all(self._data < o)) if self._data.ndim == 0 else Tensor(self._data < o)

    def __ge__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return bool(np.all(self._data >= o)) if self._data.ndim == 0 else Tensor(self._data >= o)

    def __le__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return bool(np.all(self._data <= o)) if self._data.ndim == 0 else Tensor(self._data <= o)

    # ── Arithmetic ────────────────────────────────────────────────────────
    def __add__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data + o)

    def __radd__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(o + self._data)

    def __sub__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data - o)

    def __mul__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data * o)

    def __rmul__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(o * self._data)

    def __neg__(self):
        return Tensor(-self._data)

    def __truediv__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data / o)

    # ── Comparison ────────────────────────────────────────────────────────
    def __repr__(self):
        return f"Tensor({self._data})"

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        r = self._data[idx]
        return Tensor(r) if isinstance(r, np.ndarray) else r


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────

def tensor(data, dtype=None):
    if dtype is None:
        dtype = np.float32
    return Tensor(data, dtype=dtype)


def zeros(shape, dtype=None):
    return Tensor(np.zeros(shape, dtype=dtype or np.float32))


def zeros_like(t):
    return Tensor(np.zeros_like(t._data if isinstance(t, Tensor) else t))


def ones(shape, dtype=None):
    return Tensor(np.ones(shape, dtype=dtype or np.float32))


def randn(*shape, generator=None, dtype=None):
    # Handle torch.randn((20,)) where shape is passed as a single tuple
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    if generator is not None:
        rng = np.random.RandomState(generator._seed)
        data = rng.randn(*shape)
    else:
        data = np.random.randn(*shape)
    return Tensor(data.astype(dtype or np.float32))


def randn_like(t):
    return Tensor(np.random.randn(*t.shape).astype(t.dtype if isinstance(t, Tensor) else np.float32))


def cat(tensors, dim=0):
    arrays = [t._data if isinstance(t, Tensor) else t for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim))


def stack(tensors, dim=0):
    arrays = [t._data if isinstance(t, Tensor) else t for t in tensors]
    return Tensor(np.stack(arrays, axis=dim))


# ─────────────────────────────────────────────────────────────────────────────
# Operations
# ─────────────────────────────────────────────────────────────────────────────

def norm(t, dim=None):
    data = t._data if isinstance(t, Tensor) else t
    return Tensor(np.array(np.linalg.norm(data, axis=dim)))


def abs(t):
    data = t._data if isinstance(t, Tensor) else t
    return Tensor(np.abs(data))


def max(t):
    data = t._data if isinstance(t, Tensor) else t
    return Tensor(np.array(np.max(data)))


def allclose(a, b, atol=1e-5):
    da = a._data if isinstance(a, Tensor) else a
    db = b._data if isinstance(b, Tensor) else b
    return bool(np.allclose(da, db, atol=atol))


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class Generator:
    def __init__(self, device=None):
        self._seed = 0

    def manual_seed(self, seed):
        self._seed = seed
