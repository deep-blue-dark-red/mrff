"""Type stubs for the optional native C/SIMD FWHT extension."""

from __future__ import annotations

from typing import Any

def fwht_inplace_f64(buf: Any, n_rows: int, p: int, normalized: bool) -> None: ...
def sorf_dense_f64(
    x: Any,
    inv_ls: Any,
    signs: Any,
    radii: Any,
    out: Any,
    n_samples: int,
    d: int,
    p: int,
    n_blocks: int,
    n_stacks: int,
    n_freqs: int,
) -> None: ...
def sorf_csr_f64(
    data: Any,
    indices: Any,
    indptr: Any,
    inv_ls: Any,
    signs: Any,
    radii: Any,
    out: Any,
    n_samples: int,
    d: int,
    p: int,
    n_blocks: int,
    n_stacks: int,
    n_freqs: int,
) -> None: ...
def simd_backend() -> str: ...
