# Megakernel

This repository is a development workspace for TVM/TIR experiments.  The root project only
contains tooling: the TVM source lives in `3rdparty/tvm`, and local scripts build TVM with LLVM
and CUDA enabled.

## Prerequisites

Install the system dependencies required by TVM's source build:

- Python 3.10+
- Git
- CMake 3.24+
- Ninja
- A C++17 compiler
- LLVM 15+ with `llvm-config`
- CUDA toolkit with `nvcc`
- `uv`

On a fresh checkout, initialize submodules first:

```bash
git submodule update --init --recursive
```

## Python Environment

Create or update the uv environment:

```bash
uv sync --group dev
```

## Build TVM

The default build follows TVM's official source-build flow: create `3rdparty/tvm/build`, copy
`cmake/config.cmake`, append local options, build TVM, install `tvm-ffi`, then expose
`3rdparty/tvm/python` to the uv environment with a `.pth` file. This avoids triggering a second
scikit-build CMake build during Python installation.

```bash
./scripts/build_tvm.sh
```

Useful environment variables:

```bash
CLEAN=1 ./scripts/build_tvm.sh
JOBS=32 ./scripts/build_tvm.sh
LLVM_CONFIG=/usr/lib/llvm-15/bin/llvm-config ./scripts/build_tvm.sh
CUDA_PATH=/usr/local/cuda ./scripts/build_tvm.sh
CMAKE_BUILD_TYPE=Release ./scripts/build_tvm.sh
CMAKE_GENERATOR="Unix Makefiles" ./scripts/build_tvm.sh
```

Default TVM options:

- `USE_LLVM="<detected llvm-config 15+> --ignore-libllvm --link-static"` when all static LLVM
  libs are present; otherwise the script falls back to `--link-shared` dynamic linkage.
- `USE_CUDA=ON`, or `USE_CUDA=$CUDA_PATH` when `CUDA_PATH` is set
- `USE_CUTLASS=OFF`
- `USE_CUBLAS=OFF`
- `USE_CUDNN=OFF`
- `USE_NCCL=OFF`
- `USE_NVTX=OFF`
- `USE_THRUST=OFF`
- `USE_CURAND=OFF`

## Validate

After a build, run:

```bash
./scripts/check_tvm.sh
```

The check imports `tvm` and `tvm_ffi`, prints the loaded TVM library path, and verifies that the
build reports LLVM and CUDA support.

## Reference

TVM source build documentation:
https://tvm.apache.org/docs/install/from_source.html#install-from-source
