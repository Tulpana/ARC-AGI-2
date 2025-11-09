"""Minimal NumPy compatibility shim used by the RIL stack.

The project relies on a small subset of NumPy functionality when computing
palette statistics and the temporal planar delta (Δt) masks.  The production
environment always ships with NumPy, but some CI or educational sandboxes used
for documentation tests do not.  This module exposes ``np`` and
``HAS_NUMPY``.  Callers can ``from ril.utils.safe_np import np, HAS_NUMPY`` and
continue to use the familiar ``numpy`` API.  When NumPy is unavailable we fall
back to a lightweight pure-Python implementation that supports the handful of
operations exercised by the Δt and palette gates.

The fallback intentionally implements just enough surface area for the RIL
modules.  It is *not* a drop-in replacement for real NumPy, but it does cover
array creation, boolean masking, simple reductions and a subset of logical
helpers.  When the fallback is active the module prints a short diagnostic so
logs make it clear that we are running without the native dependency.
"""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Iterable, List, Sequence, Tuple

import sys

try:  # pragma: no cover - exercised in environments with NumPy installed
    import numpy as _np  # type: ignore

    HAS_NUMPY = True
except ImportError:  # pragma: no cover - executed in restricted runtimes
    _np = None  # type: ignore
    HAS_NUMPY = False


def _normalise_dtype(dtype):
    if dtype is None:
        return None
    if dtype in (int, "int", "int64"):
        return int
    if dtype in (bool, "bool", "bool_", "boolean"):
        return bool
    if dtype in (float, "float", "float64"):
        return float
    return dtype


def _coerce_scalar(value, dtype):
    if isinstance(value, FakeArray):
        value = value.tolist()
    if dtype is None:
        return value
    try:
        return dtype(value)
    except Exception:  # pragma: no cover - best effort coercion
        return value


def _ensure_list(data, dtype=None):
    dtype = _normalise_dtype(dtype)
    if isinstance(data, FakeArray):
        return _ensure_list(data.tolist(), dtype)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [_ensure_list(item, dtype) for item in data]
    return _coerce_scalar(data, dtype)


def _shape_of(data) -> Tuple[int, ...]:
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if not data:
            return (0,)
        first_shape = _shape_of(data[0])
        return (len(data),) + first_shape
    return ()


def _broadcastable(lhs: "FakeArray", rhs: "FakeArray") -> bool:
    la = lhs.shape
    lb = rhs.shape
    if la == lb:
        return True
    if not la or not lb:
        return True
    if len(la) != len(lb):
        return False
    return all(a == b for a, b in zip(la, lb))


class FakeArray:
    """Lightweight array emulating the subset of ``numpy.ndarray`` we need."""

    def __init__(self, data, dtype=None):
        self._dtype = _normalise_dtype(dtype)
        self._data = _ensure_list(data, self._dtype)

    # ------------------------------------------------------------------
    # Basic container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.shape[0]) if self.shape else 0

    def __iter__(self):
        for item in self._data:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                yield FakeArray(item, dtype=self._dtype)
            else:
                yield item

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"FakeArray({self.tolist()!r})"

    # ------------------------------------------------------------------
    # Array shape metadata
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return _shape_of(self._data)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        shape = self.shape
        if not shape:
            return 1
        total = 1
        for dim in shape:
            total *= dim
        return total

    @property
    def dtype(self):  # pragma: no cover - used for parity with numpy
        return self._dtype or type(self._data)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def copy(self) -> "FakeArray":
        return FakeArray(self.tolist(), dtype=self._dtype)

    def tolist(self):
        def _clone(value):
            if isinstance(value, FakeArray):
                return value.tolist()
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [_clone(v) for v in value]
            return value

        return _clone(self._data)

    def astype(self, dtype):
        return FakeArray(self.tolist(), dtype=dtype)

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------
    def _resolve_indices(self, key):
        if isinstance(key, tuple):
            return key
        return (key,)

    def _index(self, data, keys):
        if not keys:
            return data
        head, *tail = keys
        if isinstance(head, slice):
            idx_range = range(len(data))[head]
            return [self._index(data[i], tuple(tail)) for i in idx_range]
        if head is Ellipsis:
            return self._index(data, tuple(tail))
        if isinstance(head, FakeArray):
            head = head.tolist()
        if isinstance(head, Sequence) and not isinstance(head, (str, bytes)):
            return [self._index(data[i], tuple(tail)) for i in head]
        return self._index(data[int(head)], tuple(tail))

    def _assign(self, data, keys, value):
        if not keys:
            return _ensure_list(value, self._dtype)
        head, *tail = keys
        if isinstance(head, slice):
            idx_range = list(range(len(data))[head])
            for offset, idx in enumerate(idx_range):
                data[idx] = self._assign(data[idx], tuple(tail), value[offset])
            return data
        if isinstance(head, FakeArray):
            head = head.tolist()
        if isinstance(head, Sequence) and not isinstance(head, (str, bytes)):
            for offset, idx in enumerate(head):
                data[int(idx)] = self._assign(data[int(idx)], tuple(tail), value[offset])
            return data
        idx = int(head)
        data[idx] = self._assign(data[idx], tuple(tail), value)
        return data

    def __getitem__(self, key):
        if isinstance(key, FakeArray) and key.shape == self.shape:
            # Boolean mask indexing – flatten values that evaluate to True
            flat = []
            mask_values = key.tolist()
            for coords in product(*[range(dim) for dim in self.shape]):
                current = mask_values
                for idx in coords:
                    current = current[idx]
                if current:
                    value = self._data
                    for idx in coords:
                        value = value[idx]
                    flat.append(value)
            return FakeArray(flat, dtype=self._dtype)
        result = self._index(self._data, self._resolve_indices(key))
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            return FakeArray(result, dtype=self._dtype)
        return result

    def __setitem__(self, key, value):
        self._data = self._assign(self._data, self._resolve_indices(key), value)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def _iter_flat(self) -> Iterable:
        def _walk(node):
            if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                for item in node:
                    yield from _walk(item)
            else:
                yield node

        yield from _walk(self._data)

    def sum(self, dtype=None):
        dtype = _normalise_dtype(dtype)
        values = [bool(v) if isinstance(v, bool) else v for v in self._iter_flat()]
        total = sum(values)
        return dtype(total) if dtype else total

    def mean(self):
        values = list(self._iter_flat())
        if not values:
            return 0.0
        return sum(values) / len(values)

    def any(self):
        return any(bool(v) for v in self._iter_flat())

    def all(self, axis=None):
        if axis is None:
            return all(bool(v) for v in self._iter_flat())
        axis = int(axis)
        if self.ndim == 1:
            return self.all()
        if self.ndim < axis + 1:
            raise ValueError("axis out of bounds")
        rows = [row.tolist() if isinstance(row, FakeArray) else row for row in self]
        if axis == 0:
            cols = list(zip(*rows)) if rows else []
            return FakeArray([all(col) for col in cols], dtype=bool)
        if axis == 1 and self.ndim >= 2:
            return FakeArray([all(row) for row in rows], dtype=bool)
        raise NotImplementedError("FakeArray.all only supports axis=0/1")

    # ------------------------------------------------------------------
    # Elementwise operations
    # ------------------------------------------------------------------
    def _binary(self, other, op, out_dtype=None):
        if isinstance(other, FakeArray):
            if not _broadcastable(self, other):
                raise ValueError("shape mismatch")
            other_data = other.tolist()
        else:
            other_data = other

        def _apply(a, b):
            if isinstance(a, Sequence) and not isinstance(a, (str, bytes)):
                if isinstance(b, Sequence) and not isinstance(b, (str, bytes)):
                    return [_apply(x, y) for x, y in zip(a, b)]
                return [_apply(x, b) for x in a]
            if isinstance(b, Sequence) and not isinstance(b, (str, bytes)):
                return [_apply(a, y) for y in b]
            return op(a, b)

        result = _apply(self.tolist(), other_data)
        return FakeArray(result, dtype=out_dtype or self._dtype)

    def __eq__(self, other):  # type: ignore[override]
        return self._binary(other, lambda a, b: a == b, out_dtype=bool)

    def __ne__(self, other):  # type: ignore[override]
        return self._binary(other, lambda a, b: a != b, out_dtype=bool)

    def __and__(self, other):
        return self._binary(other, lambda a, b: bool(a) and bool(b), out_dtype=bool)

    def __or__(self, other):
        return self._binary(other, lambda a, b: bool(a) or bool(b), out_dtype=bool)

    def __invert__(self):
        return FakeArray([[not bool(v) for v in row] for row in self.tolist()], dtype=bool)


if HAS_NUMPY:
    np = _np  # type: ignore
else:

    class _SafeNP:
        """Subset of numpy helpers backed by ``FakeArray`` objects."""

        int32 = int
        int64 = int
        float32 = float
        float64 = float
        bool_ = bool
        ndarray = FakeArray

        def array(self, data, dtype=None, copy=False):
            arr = FakeArray(data, dtype=dtype)
            return arr.copy() if copy else arr

        def asarray(self, data, dtype=None):
            return FakeArray(data, dtype=dtype)

        def empty(self, shape, dtype=None):
            return self.zeros(shape, dtype=dtype)

        def zeros(self, shape, dtype=None):
            dtype = _normalise_dtype(dtype)
            if isinstance(shape, int):
                shape = (shape,)

            def _build(dimensions):
                if not dimensions:
                    return _coerce_scalar(0, dtype)
                length = dimensions[0]
                return [_build(dimensions[1:]) for _ in range(length)]

            return FakeArray(_build(tuple(shape)), dtype=dtype)

        def ones(self, shape, dtype=None):
            dtype = _normalise_dtype(dtype)
            if isinstance(shape, int):
                shape = (shape,)

            def _build(dimensions):
                if not dimensions:
                    return _coerce_scalar(1, dtype)
                length = dimensions[0]
                return [_build(dimensions[1:]) for _ in range(length)]

            return FakeArray(_build(tuple(shape)), dtype=dtype)

        def zeros_like(self, arr, dtype=None):
            arr = FakeArray(arr)
            return self.zeros(arr.shape, dtype=dtype)

        def full_like(self, arr, fill_value, dtype=None):
            return self.full(FakeArray(arr).shape, fill_value, dtype=dtype)

        def ones_like(self, arr, dtype=None):
            arr = FakeArray(arr)
            return self.ones(arr.shape, dtype=dtype)

        def full(self, shape, fill_value, dtype=None):
            dtype = _normalise_dtype(dtype)
            if isinstance(shape, int):
                shape = (shape,)

            def _build(dimensions):
                if not dimensions:
                    return _coerce_scalar(fill_value, dtype)
                length = dimensions[0]
                return [_build(dimensions[1:]) for _ in range(length)]

            return FakeArray(_build(tuple(shape)), dtype=dtype)

        def any(self, arr):
            return FakeArray(arr).any()

        def all(self, arr, axis=None):
            return FakeArray(arr).all(axis=axis)

        def mean(self, arr):
            if isinstance(arr, Sequence) and not isinstance(arr, FakeArray):
                return sum(arr) / len(arr) if arr else 0.0
            return FakeArray(arr).mean()

        def sum(self, arr, dtype=None):
            return FakeArray(arr).sum(dtype=dtype)

        def where(self, condition, x, y):
            cond = FakeArray(condition)
            xa = FakeArray(x)
            ya = FakeArray(y)

            def _walk(cond_node, x_node, y_node):
                if not isinstance(cond_node, Sequence) or isinstance(cond_node, (str, bytes)):
                    return x_node if cond_node else y_node
                length = len(cond_node)
                out = []
                for idx in range(length):
                    out.append(_walk(cond_node[idx], x_node[idx], y_node[idx]))
                return out

            return FakeArray(_walk(cond.tolist(), xa.tolist(), ya.tolist()))

        def unique(self, arr, return_counts=False):
            values = list(FakeArray(arr)._iter_flat())
            counts = Counter(values)
            uniques = sorted(counts.keys())
            values_array = FakeArray(uniques)
            if not return_counts:
                return values_array
            count_array = FakeArray([counts[value] for value in uniques])
            return values_array, count_array

        def logical_and(self, a, b):
            return FakeArray(a) & FakeArray(b)

        def logical_or(self, a, b):
            return FakeArray(a) | FakeArray(b)

        def equal(self, a, b):
            return FakeArray(a) == FakeArray(b)

        def isin(self, element, test_elements):
            arr = FakeArray(element)
            tests = set(int(v) for v in test_elements)

            def _check(node):
                if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                    return [_check(child) for child in node]
                return int(node) in tests

            return FakeArray(_check(arr.tolist()), dtype=bool)

        def argmax(self, arr):
            data = list(FakeArray(arr)._iter_flat())
            if not data:
                return 0
            max_value = max(data)
            return data.index(max_value)

        def nonzero(self, arr):
            array = FakeArray(arr)
            shape = array.shape
            if len(shape) == 1:
                indices = [idx for idx, value in enumerate(array.tolist()) if value]
                return (indices,)
            if len(shape) == 2:
                ys = []
                xs = []
                for y, row in enumerate(array.tolist()):
                    for x, value in enumerate(row):
                        if value:
                            ys.append(y)
                            xs.append(x)
                return ys, xs
            raise NotImplementedError("nonzero only implemented for 1D/2D arrays")


    np = _SafeNP()


if not HAS_NUMPY:
    msg = "[Δt] safe-np active: running in pure Python fallback mode (NumPy unavailable)"
    print(msg, file=sys.stderr)


__all__ = ["np", "HAS_NUMPY", "FakeArray"]

