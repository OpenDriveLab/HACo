"""Msgpack helpers for the workstation deployment wire format."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


def _encode_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported numpy dtype for msgpack: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(order="C"),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _lookup(obj: dict[Any, Any], key: bytes) -> Any:
    if key in obj:
        return obj[key]
    return obj[key.decode("ascii")]


def _has_key(obj: dict[Any, Any], key: bytes) -> bool:
    return key in obj or key.decode("ascii") in obj


def _unpack_dtype(dtype: Any) -> np.dtype:
    if isinstance(dtype, bytes):
        dtype = dtype.decode("ascii")
    if isinstance(dtype, (list, tuple)):
        dtype = [
            (
                item[0].decode("ascii")
                if isinstance(item[0], bytes)
                else item[0],
                _unpack_dtype(item[1]),
            )
            + tuple(item[2:])
            for item in dtype
        ]
    return np.dtype(dtype)


def _decode_numpy(obj: dict[Any, Any]) -> Any:
    try:
        if _has_key(obj, b"__ndarray__"):
            return np.ndarray(
                buffer=_lookup(obj, b"data"),
                dtype=np.dtype(_lookup(obj, b"dtype")),
                shape=tuple(_lookup(obj, b"shape")),
            ).copy()
        if _has_key(obj, b"__npgeneric__"):
            return np.dtype(_lookup(obj, b"dtype")).type(_lookup(obj, b"data"))
        if _has_key(obj, b"nd"):
            if bool(_lookup(obj, b"nd")):
                dtype = _unpack_dtype(_lookup(obj, b"type"))
                return np.ndarray(
                    buffer=_lookup(obj, b"data"),
                    dtype=dtype,
                    shape=tuple(_lookup(obj, b"shape")),
                ).copy()
            dtype = _unpack_dtype(_lookup(obj, b"type"))
            return np.frombuffer(_lookup(obj, b"data"), dtype=dtype)[0]
    except (KeyError, TypeError, ValueError):
        return obj
    return obj


def packb(payload: Any) -> bytes:
    return msgpack.packb(
        payload,
        default=_encode_numpy,
        use_bin_type=True,
        strict_types=False,
    )


def unpackb(payload: bytes) -> Any:
    kwargs = {
        "object_hook": _decode_numpy,
        "raw": False,
        "strict_map_key": False,
    }
    try:
        return msgpack.unpackb(payload, max_buffer_size=256 * 1024 * 1024, **kwargs)
    except TypeError:
        return msgpack.unpackb(payload, **kwargs)
