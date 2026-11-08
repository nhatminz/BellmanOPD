#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3}")"
# Use the node's compiler/runtime pair by default, with explicit overrides.
export CC="${ALFWORLD_CC:-/usr/bin/gcc}"
export CXX="${ALFWORLD_CXX:-/usr/bin/g++}"
export ALFWORLD_BUILD_JOBS="${ALFWORLD_BUILD_JOBS:-2}"
if ! [[ "${ALFWORLD_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ALFWORLD_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi
for tool in "${CC}" "${CXX}" cmake make; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Missing build tool: ${tool}. Install gcc, g++, make and cmake on this node, or set ALFWORLD_CC/ALFWORLD_CXX." >&2
    exit 2
  fi
done
NATIVE_SITE="$("${PYTHON_BIN}" -m b200_experiment.alfworld_env --native-site)"
if [[ -e "${NATIVE_SITE}" && ! -L "${NATIVE_SITE}" ]]; then
  echo "Refusing to replace an existing directory: ${NATIVE_SITE}. Set ALFWORLD_NATIVE_SITE to a new path." >&2
  exit 2
fi
mkdir -p "$(dirname "${NATIVE_SITE}")"
BUILD_ROOT="$(mktemp -d "$(dirname "${NATIVE_SITE}")/.build-XXXXXXXX")"
PUBLISH_LINK="${NATIVE_SITE}.link-$$"
cleanup() {
  rm -f -- "${PUBLISH_LINK}"
  if [[ -n "${BUILD_ROOT}" ]]; then rm -rf -- "${BUILD_ROOT}"; fi
}
trap cleanup EXIT
mkdir -p "${BUILD_ROOT}/tools"
export ALFWORLD_BUILD_PYTHON="${PYTHON_BIN}"
export ALFWORLD_BUILD_MAKE="$(command -v make)"
# Upstream setup.py invokes literal `python`; build.py invokes `make -j<NCPU>`.
# Keep both inside this build and bound parallelism even on large B200 nodes.
cat > "${BUILD_ROOT}/tools/python" <<'SH'
#!/usr/bin/env bash
exec "${ALFWORLD_BUILD_PYTHON}" "$@"
SH
cat > "${BUILD_ROOT}/tools/make" <<'SH'
#!/usr/bin/env bash
exec "${ALFWORLD_BUILD_MAKE}" "$@" "-j${ALFWORLD_BUILD_JOBS}"
SH
chmod +x "${BUILD_ROOT}/tools/python" "${BUILD_ROOT}/tools/make"
export PATH="${BUILD_ROOT}/tools:$(dirname "${PYTHON_BIN}"):${PATH}"
FD_VERSION="$("${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
try:
    print(version("fast-downward-textworld"))
except PackageNotFoundError:
    print("20.6.4")
PY
)"
echo "Building Fast Downward ${FD_VERSION} with ${CXX} for $(hostname)"
echo "Isolated package target: ${NATIVE_SITE} (shared venv stays unchanged)"
"${PYTHON_BIN}" -m pip install --no-deps --no-cache-dir \
  --no-binary=fast-downward-textworld \
  --target "${BUILD_ROOT}/site-packages" "fast-downward-textworld==${FD_VERSION}"
# Probe in a fresh Python process BEFORE publishing the build. This catches
# GLIBCXX/CXXABI errors in the actual dlopen path, without importing torch.
PYTHONPATH="${BUILD_ROOT}/site-packages:${PYTHONPATH}" "${PYTHON_BIN}" - <<'PY'
import fast_downward
library = fast_downward.load_lib()
fast_downward.close_lib(library)
print("Native library load passed:", fast_downward.__file__)
PY
ln -s "$(basename "${BUILD_ROOT}")/site-packages" "${PUBLISH_LINK}"
mv -Tf -- "${PUBLISH_LINK}" "${NATIVE_SITE}"
# Retain this version and older versions for any live environment workers.
BUILD_ROOT=""
echo "Installed isolated native runtime: ${NATIVE_SITE}"
echo "Next: ${PYTHON_BIN} -m b200_experiment.alfworld_cli smoke"
