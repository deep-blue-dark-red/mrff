"""Build configuration for optional native C/SIMD FWHT and sparse SORF extension."""

from __future__ import annotations

import os
import sys

from setuptools import Extension, setup

extra_compile_args: list[str] = []
if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3"]

ext_modules = []
if os.environ.get("MRFF_PURE_PYTHON", "0") != "1":
    ext_modules.append(
        Extension(
            name="mrff._fwht_c",
            sources=["src/mrff/_fwht_c.c"],
            extra_compile_args=extra_compile_args,
            optional=True,
        )
    )

setup(ext_modules=ext_modules)
