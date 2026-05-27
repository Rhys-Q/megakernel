#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "error: command failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TVM_BUILD_DIR="${PROJECT_ROOT}/3rdparty/tvm/build"

export TVM_LIBRARY_PATH="${TVM_BUILD_DIR}"

cd "${PROJECT_ROOT}"

uv run python - <<'PY'
import sys

import tvm
import tvm_ffi
from tvm import support
from tvm.base import _LIB
from tvm.target import Target

info = support.libinfo()

print(f"python: {sys.executable}")
print(f"tvm: {tvm.__file__}")
print(f"tvm_ffi: {tvm_ffi.__file__}")
print(f"tvm library: {getattr(_LIB, '_name', _LIB)}")
print(f"TVM_LIBRARY_PATH: {info.get('TVM_LIBRARY_PATH', '<unknown>')}")

for key in ("USE_LLVM", "LLVM_VERSION", "USE_CUDA", "CUDA_VERSION", "HIDE_PRIVATE_SYMBOLS"):
    print(f"{key}: {info.get(key, '<missing>')}")

if str(info.get("USE_LLVM", "")).upper() not in {"ON", "TRUE", "1"}:
    raise SystemExit("TVM was not built with LLVM enabled")

if str(info.get("USE_CUDA", "")).upper() not in {"ON", "TRUE", "1"}:
    raise SystemExit("TVM was not built with CUDA enabled")

print(f"cuda target: {Target('cuda')}")
print(f"cuda device exists: {tvm.cuda().exist}")
PY
