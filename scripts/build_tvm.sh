#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "error: command failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TVM_DIR="${PROJECT_ROOT}/3rdparty/tvm"
TVM_BUILD_DIR="${TVM_DIR}/build"
TVM_FFI_DIR="${TVM_DIR}/3rdparty/tvm-ffi"

detect_default_llvm_config() {
  local candidate
  local resolved
  local version
  local major
  for candidate in \
    llvm-config \
    llvm-config-21 \
    llvm-config-20 \
    llvm-config-19 \
    llvm-config-18 \
    llvm-config-17 \
    llvm-config-16 \
    llvm-config-15 \
    /usr/lib/llvm-21/bin/llvm-config \
    /usr/lib/llvm-20/bin/llvm-config \
    /usr/lib/llvm-19/bin/llvm-config \
    /usr/lib/llvm-18/bin/llvm-config \
    /usr/lib/llvm-17/bin/llvm-config \
    /usr/lib/llvm-16/bin/llvm-config \
    /usr/lib/llvm-15/bin/llvm-config; do
    if resolved="$(command -v "${candidate}" 2>/dev/null)"; then
      version="$("${resolved}" --version 2>/dev/null || true)"
      major="${version%%.*}"
      if [[ "${major}" =~ ^[0-9]+$ && "${major}" -ge 15 ]]; then
        echo "${resolved}"
        return
      fi
    fi
  done
  echo "llvm-config"
}

CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
CMAKE_GENERATOR_NAME="${CMAKE_GENERATOR:-Ninja}"
if [[ -n "${LLVM_CONFIG:-}" ]]; then
  LLVM_CONFIG_VALUE="${LLVM_CONFIG}"
  LLVM_CONFIG_BIN="${LLVM_CONFIG_VALUE%% *}"
else
  LLVM_CONFIG_BIN="$(detect_default_llvm_config)"
  LLVM_CONFIG_VALUE=""
fi

if [[ -n "${CUDA_PATH:-}" ]]; then
  USE_CUDA_VALUE="${CUDA_PATH}"
  export PATH="${CUDA_PATH}/bin:${PATH}"
  export CUDACXX="${CUDA_PATH}/bin/nvcc"
else
  USE_CUDA_VALUE="ON"
fi

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  else
    echo 1
  fi
}

JOBS="${JOBS:-$(detect_jobs)}"

require_cmd() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "error: required command not found: ${name}" >&2
    exit 127
  fi
}

check_llvm_version() {
  local version
  local major
  version="$("${LLVM_CONFIG_BIN}" --version)"
  major="${version%%.*}"
  if [[ ! "${major}" =~ ^[0-9]+$ || "${major}" -lt 15 ]]; then
    echo "error: LLVM >= 15 is required, but ${LLVM_CONFIG_BIN} reports ${version}" >&2
    echo "hint: set LLVM_CONFIG=/path/to/llvm-config if multiple LLVM versions are installed" >&2
    exit 1
  fi
}

llvm_static_link_is_usable() {
  local libfile
  while IFS= read -r libfile; do
    [[ -z "${libfile}" ]] && continue
    if [[ ! -e "${libfile}" ]]; then
      echo "warning: LLVM static library is missing: ${libfile}" >&2
      return 1
    fi
  done < <("${LLVM_CONFIG_BIN}" --ignore-libllvm --link-static --libfiles | tr ' ' '\n')
}

resolve_llvm_config_value() {
  if [[ -n "${LLVM_CONFIG_VALUE}" ]]; then
    return
  fi

  if llvm_static_link_is_usable; then
    LLVM_CONFIG_VALUE="${LLVM_CONFIG_BIN} --ignore-libllvm --link-static"
  else
    LLVM_CONFIG_VALUE="${LLVM_CONFIG_BIN} --link-shared"
    echo "warning: falling back to dynamic LLVM linkage: ${LLVM_CONFIG_VALUE}" >&2
  fi
}

write_tvm_config() {
  mkdir -p "${TVM_BUILD_DIR}"
  cp "${TVM_DIR}/cmake/config.cmake" "${TVM_BUILD_DIR}/config.cmake"
  cat >>"${TVM_BUILD_DIR}/config.cmake" <<EOF

# Local megakernel build options.
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
set(USE_LLVM "${LLVM_CONFIG_VALUE}")
set(HIDE_PRIVATE_SYMBOLS ON)

set(USE_CUDA "${USE_CUDA_VALUE}")
set(USE_CUBLAS OFF)
set(USE_CUDNN OFF)
set(USE_CUDNN_FRONTEND OFF)
set(USE_CUTLASS OFF)
set(USE_NCCL OFF)
set(USE_NVTX OFF)
set(USE_THRUST OFF)
set(USE_CURAND OFF)

set(USE_OPENCL OFF)
set(USE_VULKAN OFF)
set(USE_METAL OFF)
set(USE_ROCM OFF)
set(USE_RCCL OFF)
set(USE_TENSORRT_CODEGEN OFF)
set(USE_TENSORRT_RUNTIME OFF)
EOF
}

sanitize_tvm_build_dir() {
  local cache="${TVM_BUILD_DIR}/CMakeCache.txt"
  local generator=""
  local make_program=""

  [[ -f "${cache}" ]] || return

  generator="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${cache}" | head -n 1)"
  make_program="$(sed -n 's/^CMAKE_MAKE_PROGRAM:[^=]*=//p' "${cache}" | head -n 1)"

  if [[ -f "${TVM_BUILD_DIR}/.skbuild-info.json" ]] || rg -q '^SKBUILD:' "${cache}"; then
    echo "warning: removing TVM build directory contaminated by scikit-build metadata"
    rm -rf "${TVM_BUILD_DIR}"
    return
  fi

  if [[ "${generator}" == "Unix Makefiles" && "${make_program}" == *ninja* ]]; then
    echo "warning: removing TVM build directory with mismatched generator/make program"
    echo "warning: generator=${generator}, CMAKE_MAKE_PROGRAM=${make_program}"
    rm -rf "${TVM_BUILD_DIR}"
    return
  fi

  if [[ "${generator}" == "Ninja" && ! -f "${TVM_BUILD_DIR}/build.ninja" ]]; then
    echo "warning: removing TVM build directory with Ninja cache but missing build.ninja"
    rm -rf "${TVM_BUILD_DIR}"
  fi
}

configure_tvm() {
  local cmake_args
  cmake_args=(-S "${TVM_DIR}" -B "${TVM_BUILD_DIR}")

  if [[ -f "${TVM_BUILD_DIR}/CMakeCache.txt" ]]; then
    local cached_generator
    cached_generator="$(
      sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${TVM_BUILD_DIR}/CMakeCache.txt" | head -n 1
    )"
    if [[ -n "${cached_generator}" ]]; then
      echo "Reusing existing CMake generator: ${cached_generator}"
    fi
  else
    cmake_args+=(-G "${CMAKE_GENERATOR_NAME}")
  fi

  cmake "${cmake_args[@]}"
}

install_tvm_python_path() {
  local tvm_python_dir="${TVM_DIR}/python"

  uv run python - "${tvm_python_dir}" <<'PY'
import site
import sys
from pathlib import Path

tvm_python_dir = Path(sys.argv[1]).resolve()
if not tvm_python_dir.is_dir():
    raise SystemExit(f"TVM python directory not found: {tvm_python_dir}")

site_packages = next((Path(p) for p in site.getsitepackages() if Path(p).is_dir()), None)
if site_packages is None:
    site_packages = Path(site.getusersitepackages())
    site_packages.mkdir(parents=True, exist_ok=True)

pth = site_packages / "megakernel-tvm-dev.pth"
pth.write_text(f"{tvm_python_dir}\n", encoding="utf-8")
print(f"Installed TVM Python path: {pth} -> {tvm_python_dir}")
PY
}

main() {
  cd "${PROJECT_ROOT}"

  require_cmd uv
  require_cmd git
  require_cmd cmake
  require_cmd ninja
  require_cmd python
  require_cmd "${LLVM_CONFIG_BIN}"
  require_cmd nvcc
  check_llvm_version
  resolve_llvm_config_value
  echo "Using LLVM config: ${LLVM_CONFIG_VALUE}"

  if [[ ! -d "${TVM_DIR}" ]]; then
    echo "error: TVM submodule not found at ${TVM_DIR}" >&2
    exit 1
  fi

  git submodule update --init --recursive

  uv sync --group dev

  if [[ "${CLEAN:-0}" == "1" ]]; then
    rm -rf "${TVM_BUILD_DIR}"
  fi

  sanitize_tvm_build_dir
  write_tvm_config

  configure_tvm
  cmake --build "${TVM_BUILD_DIR}" --parallel "${JOBS}"

  uv pip install --reinstall -v "${TVM_FFI_DIR}"
  install_tvm_python_path

  echo "TVM build completed."
  echo "TVM_LIBRARY_PATH=${TVM_BUILD_DIR}"
}

main "$@"
