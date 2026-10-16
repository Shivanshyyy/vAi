#!/usr/bin/env bash
set -u

# Vulkan / llama.cpp build dependency installer for Ubuntu/Debian
# Installs required packages and verifies packages, commands, headers and libraries.

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "[ERROR] sudo is required when not running as root."
    exit 1
  fi
  SUDO="sudo"
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "[ERROR] This script supports Ubuntu/Debian systems using apt-get."
  exit 1
fi

PACKAGES=(
  build-essential
  cmake
  ninja-build
  libvulkan-dev
  vulkan-tools
  glslc
  spirv-headers
)

echo "=============================================================="
echo " Vulkan build dependency installer"
echo " Ubuntu / Debian"
echo "=============================================================="

echo
printf '[INFO] OS: '
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  echo "${PRETTY_NAME:-unknown}"
else
  echo "unknown"
fi

echo
 echo "[1/4] Updating apt package index..."
if ! $SUDO apt-get update; then
  echo "[ERROR] apt-get update failed."
  exit 1
fi

echo
 echo "[2/4] Installing packages..."
INSTALL_FAILED=0
for pkg in "${PACKAGES[@]}"; do
  echo "--------------------------------------------------------------"
  echo "[INSTALL] ${pkg}"
  if $SUDO apt-get install -y "$pkg"; then
    echo "[OK] apt install ${pkg}"
  else
    echo "[FAIL] apt install ${pkg}"
    INSTALL_FAILED=1
  fi
done

echo
 echo "[3/4] Verifying installed packages..."
VERIFY_FAILED=0
for pkg in "${PACKAGES[@]}"; do
  if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q '^install ok installed$'; then
    version=$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null || true)
    printf '[OK]   %-20s %s\n' "$pkg" "$version"
  else
    printf '[FAIL] %-20s not installed\n' "$pkg"
    VERIFY_FAILED=1
  fi
done

echo
 echo "[4/4] Verifying Vulkan build environment..."

check_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    printf '[OK]   command %-12s -> %s\n' "$cmd" "$(command -v "$cmd")"
    return 0
  else
    printf '[FAIL] command %-12s not found\n' "$cmd"
    VERIFY_FAILED=1
    return 1
  fi
}

check_file() {
  local path="$1"
  local desc="$2"
  if [[ -e "$path" ]]; then
    printf '[OK]   %-28s %s\n' "$desc" "$path"
    return 0
  else
    printf '[FAIL] %-28s %s not found\n' "$desc" "$path"
    VERIFY_FAILED=1
    return 1
  fi
}

check_cmd cmake
check_cmd ninja
check_cmd glslc
check_cmd vulkaninfo

# Common Ubuntu/Debian x86_64 locations
check_file /usr/include/vulkan/vulkan.h "Vulkan header"

if [[ -e /usr/lib/x86_64-linux-gnu/libvulkan.so ]]; then
  echo "[OK]   Vulkan library               /usr/lib/x86_64-linux-gnu/libvulkan.so"
elif ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so'; then
  echo "[OK]   Vulkan library               found via ldconfig"
else
  echo "[FAIL] Vulkan library               libvulkan.so not found"
  VERIFY_FAILED=1
fi

# SPIR-V header / CMake package checks
if dpkg -L spirv-headers 2>/dev/null | grep -q '/spirv/'; then
  echo "[OK]   SPIR-V headers               installed"
else
  echo "[FAIL] SPIR-V headers               not found"
  VERIFY_FAILED=1
fi

SPIRV_CMAKE=$(dpkg -L spirv-headers 2>/dev/null | grep -E '/cmake/.+(SPIRV|spirv).*(Config|Targets).*\.cmake$' | head -n 1 || true)
if [[ -n "$SPIRV_CMAKE" ]]; then
  echo "[OK]   SPIR-V CMake package         $SPIRV_CMAKE"
else
  echo "[WARN] SPIR-V CMake config          no Config/Targets cmake file detected via dpkg -L"
fi

echo
 echo "-------------------- Tool versions ---------------------------"
command -v cmake >/dev/null 2>&1 && cmake --version | head -n 1
command -v ninja >/dev/null 2>&1 && ninja --version | sed 's/^/ninja version /'
command -v glslc >/dev/null 2>&1 && glslc --version | head -n 1

echo
 echo "-------------------- Vulkan runtime ---------------------------"
if command -v vulkaninfo >/dev/null 2>&1; then
  VULKAN_TMP=$(mktemp)
  if vulkaninfo --summary >"$VULKAN_TMP" 2>&1; then
    echo "[OK] vulkaninfo --summary succeeded"
    cat "$VULKAN_TMP"
  else
    echo "[WARN] Vulkan packages are installed, but vulkaninfo could not initialize Vulkan."
    echo "       This usually means the GPU Vulkan driver/runtime is missing or unavailable"
    echo "       (for example inside a container/WSL without GPU passthrough)."
    echo
    tail -n 30 "$VULKAN_TMP"
  fi
  rm -f "$VULKAN_TMP"
fi

echo
 echo "=============================================================="
if [[ "$INSTALL_FAILED" -eq 0 && "$VERIFY_FAILED" -eq 0 ]]; then
  echo " RESULT: SUCCESS"
  echo " All required Vulkan build packages/tools were installed and verified."
  echo "=============================================================="
  exit 0
else
  echo " RESULT: FAILED / INCOMPLETE"
  [[ "$INSTALL_FAILED" -ne 0 ]] && echo " - One or more package installations failed."
  [[ "$VERIFY_FAILED" -ne 0 ]] && echo " - One or more required components could not be verified."
  echo "=============================================================="
  exit 1
fi
