#!/usr/bin/env bash
#==============================================================================
# Installs the llama.cpp the benches compare against: pinned tag, Vulkan
# backend, static by default (no rpath or LD_LIBRARY_PATH needed; --shared
# installs shared libraries with an $ORIGIN rpath instead).
#
# Nothing is written into the repository. The build goes to the user cache,
#   ${XDG_CACHE_HOME:-~/.cache}/vai/llama.cpp-<tag>
# and is registered in CMake's user package registry (~/.cmake/packages), so
# find_package(llama) in any checkout finds it without flags. The GGUF
# converter from the same tag (convert_hf_to_gguf.py + conversion/ + gguf-py
# + requirements, MIT) is kept at <prefix>/convert so the benches' GGUFs come
# from the exact tag the runtime was built at — see benchmarks/README.md,
# "GGUF Conversion".
#
# Needs git, cmake, glslc and the SPIRV-Headers CMake package. The LunarG
# Vulkan SDK carries the last two, a distro Vulkan may not (Ubuntu: apt
# install glslc spirv-headers). Both are checked before anything is built.
#
#   ./benchmarks/scripts/setup-llamacpp.sh                 # ~2-3 min, then build vAi
#   ./benchmarks/scripts/setup-llamacpp.sh --shared --keep-src
#   ./benchmarks/scripts/setup-llamacpp.sh --uninstall     # remove build + registration
#
# To use your own llama.cpp build instead, skip this and configure vAi with
# -Dllama_DIR=<prefix>/lib/cmake/llama; that takes precedence over the registry.
#==============================================================================
set -euo pipefail

# The benches are calibrated against this build; benchmarks/*/CMakeLists.txt warns
# when it finds a different one, because the numbers stop being comparable.
TAG=b10426
BUILD_NUMBER="${TAG#b}"

[ -n "${HOME:-}" ] || { echo "HOME is not set — the registry lives under \$HOME/.cmake; set it and re-run" >&2; exit 2; }
if [ "$(id -u)" -eq 0 ]; then
    echo "warning: running as root — this installs into root's cache ($HOME), not yours" >&2
fi

# The script owns exactly these: the prefix, its source tree, and the registry
# entries it wrote (vai-*). It deletes nothing else.
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
PREFIX="$CACHE_DIR/vai/llama.cpp-$TAG"
SRC="$PREFIX-src"
CONFIG="$PREFIX/lib/cmake/llama/llama-config.cmake"
REG_DIR="$HOME/.cmake/packages/llama"

JOBS="$(nproc 2>/dev/null || echo 4)"
SHARED=OFF
KEEP_SRC=0
FORCE=0
UNINSTALL=0

usage() {
    sed -n '2,/^#===/p' "$0" | sed 's/^#//'
    cat <<USAGE

Options:
  --jobs N, -j N parallel build jobs (default: $JOBS)
  --shared       shared libraries instead of static
  --keep-src     keep the source and build trees (~600 MB) for re-runs
  --force        reinstall even if this tag is already installed
  --uninstall    remove the build, its source tree and the registry entry
  -h, --help     this text
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --jobs|-j)   JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        --shared)    SHARED=ON;    shift ;;
        --keep-src)  KEEP_SRC=1;   shift ;;
        --force)     FORCE=1;      shift ;;
        --uninstall) UNINSTALL=1;  shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs needs a positive integer, got '$JOBS'" >&2; exit 2; }

variantName() {
    case "$1" in
        ON)  echo shared ;;
        OFF) echo static ;;
        *)   echo "unknown linkage" ;;
    esac
}
VARIANT="$(variantName $SHARED)"

# CMake's user package registry: one file per entry, its content the directory
# holding llama-config.cmake. find_package consults it after CMAKE_PREFIX_PATH,
# HINTS and $PATH-derived prefixes (the benches turn those off) but before
# system prefixes, so -Dllama_DIR still wins. Only one
# vai-* entry is kept: with several, which one find_package picks is undefined.
register() {
    mkdir -p "$REG_DIR"
    rm -f "$REG_DIR"/vai-*
    printf '%s\n' "$PREFIX/lib/cmake/llama" > "$REG_DIR/vai-$TAG"
}

# Keep the GGUF converter next to the install so it survives the source
# cleanup. It is llama.cpp's own script (MIT); LICENSE rides along with it.
# Not self-contained: it imports the sibling conversion/ package, and its
# requirements file pulls in requirements-convert_legacy_llama.txt.
CONVERT="$PREFIX/convert"
keep_converter() {
    mkdir -p "$CONVERT"
    cp -R "$SRC/convert_hf_to_gguf.py" "$SRC/conversion" "$SRC/gguf-py" "$SRC/LICENSE" "$CONVERT/"
    cp "$SRC/requirements/requirements-convert_hf_to_gguf.txt" \
       "$SRC/requirements/requirements-convert_legacy_llama.txt" "$CONVERT/"
}

# All of it: an install left incomplete by an older run is repaired instead of
# passing as "already installed" and failing at conversion time.
converterComplete() {
    [ -f "$CONVERT/convert_hf_to_gguf.py" ] \
        && [ -d "$CONVERT/conversion" ] \
        && [ -d "$CONVERT/gguf-py" ] \
        && [ -f "$CONVERT/requirements-convert_hf_to_gguf.txt" ] \
        && [ -f "$CONVERT/requirements-convert_legacy_llama.txt" ]
}

if [ $UNINSTALL -eq 1 ]; then
    rm -rf "$PREFIX" "$SRC"
    rm -f "$REG_DIR"/vai-*
    echo "removed $PREFIX, $SRC and the vai-* entries in $REG_DIR"
    exit 0
fi

#------------------------------------------------------------------------------
# Already installed?
#
# The prefix carries the tag but not the linkage, so read both back out of the
# installed config anyway (a hand-made install could be anything). Either one
# differing means a rebuild: the benches only compare against $TAG, and mixing
# the two linkages in one prefix leaves llama-config.cmake's find_library() to
# choose between a leftover .so and the new .a — it prefers the .so. That is
# why a rebuild wipes the prefix (just before install, below) instead of
# installing over it.
#------------------------------------------------------------------------------
if [ -f "$CONFIG" ]; then
    installed="$(sed -n 's/^set(LLAMA_BUILD_NUMBER *\([0-9]*\)).*/\1/p' "$CONFIG" | head -1)"
    installedShared="$(sed -n 's/^set(LLAMA_SHARED_LIB *\([A-Za-z]*\)).*/\1/p' "$CONFIG" | head -1)"

    if [ $FORCE -eq 0 ] \
       && [ "$installed" = "$BUILD_NUMBER" ] \
       && [ "$installedShared" = "$SHARED" ] \
       && [ -f "$PREFIX/include/mtmd.h" ]; then
        register    # restore the entry if it was removed by hand
        echo "llama.cpp $TAG ($VARIANT) is already installed at $PREFIX"
        # A converter that is missing, or kept only in part by an older run:
        # complete it from a --keep-src tree if one is around.
        if ! converterComplete; then
            if [ -f "$SRC/convert_hf_to_gguf.py" ] && [ -d "$SRC/conversion" ]; then
                keep_converter
                echo "completed the GGUF converter at $CONVERT"
            else
                echo "note: the GGUF converter at $CONVERT is missing or incomplete"
            fi
        fi
        echo "(re-run with --force to rebuild it)"
        exit 0
    fi
fi

#------------------------------------------------------------------------------
# Prerequisites — checked before anything is cloned, built or deleted, so a
# missing one is not a build error 2 minutes in. Vulkan itself is left to the
# real configure.
#------------------------------------------------------------------------------
missing=""
for tool in git cmake glslc; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
    echo "missing required tool(s):$missing" >&2
    echo "  glslc ships with the Vulkan SDK (Ubuntu: apt install glslc, or the LunarG SDK)" >&2
    exit 1
fi

# The one prerequisite that is not a tool: ggml-vulkan does
# find_package(SPIRV-Headers), which libvulkan-dev + glslc alone do not pull in.
# Probing configures an empty project, so any copy cmake can reach counts;
# `cmake -P` cannot stand in, as the config defines a target.
hasSpirvHeaders() {
    local dir rc=0
    dir="$(mktemp -d)"
    cat > "$dir/CMakeLists.txt" <<'PROBE'
cmake_minimum_required(VERSION 3.16)
project(spirv-headers-probe NONE)
find_package(SPIRV-Headers REQUIRED)
PROBE
    cmake -S "$dir" -B "$dir/build" >/dev/null 2>&1 || rc=1
    rm -rf "$dir"
    return $rc
}

if ! hasSpirvHeaders; then
    cat >&2 <<'MISSING'
missing the SPIRV-Headers CMake package — the Vulkan backend requires it, and
libvulkan-dev/glslc alone do not provide it:
  Ubuntu/Debian   apt install spirv-headers
  Fedora          dnf install spirv-headers-devel
  Arch            pacman -S spirv-headers
  or install the LunarG Vulkan SDK, which ships it
MISSING
    exit 1
fi

GENERATOR=()
command -v ninja >/dev/null 2>&1 && GENERATOR=(-G Ninja)

if [ -f "$CONFIG" ]; then
    echo "replacing llama.cpp build ${installed:-unknown} ($(variantName "${installedShared:-}")) at $PREFIX"
fi

echo "=============================================================="
echo " llama.cpp $TAG -> $PREFIX"
echo " $VARIANT | Vulkan | $JOBS jobs"
echo "=============================================================="

#------------------------------------------------------------------------------
# Source
#
# A tree kept by --keep-src is reused only when it really is at $TAG. The build
# number is stamped on from $BUILD_NUMBER below rather than derived from the
# checkout, so a tree left behind by an earlier $TAG would otherwise be
# installed under the new number — passing the check above and the version
# warning in the benches while holding the old code. After a $TAG bump the old
# tree has no ref for the new tag, so the lookup fails and it is re-cloned.
#------------------------------------------------------------------------------
mkdir -p "$CACHE_DIR/vai"

srcTag="$(git -C "$SRC" rev-parse -q --verify "refs/tags/$TAG^{commit}" 2>/dev/null || true)"
srcHead="$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)"

if [ -n "$srcTag" ] && [ "$srcTag" = "$srcHead" ]; then
    echo "[1/3] reusing source at $SRC"
else
    if [ -d "$SRC" ]; then
        echo "[1/3] source at $SRC is not $TAG — re-cloning..."
    else
        echo "[1/3] cloning $TAG..."
    fi
    rm -rf "$SRC"
    # Ctrl-C mid-clone or mid-build: drop the half-made tree, unless it was
    # asked for. A reused tree is never touched. The existing install is not at
    # risk here — it is replaced only after the build succeeds, below.
    [ $KEEP_SRC -eq 1 ] || { trap 'rm -rf "$SRC"; exit 130' INT; trap 'rm -rf "$SRC"; exit 143' TERM; }
    git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi

#------------------------------------------------------------------------------
# Build + install
#
# LLAMA_BUILD_APP=OFF: at this tag the unified binary does not build once the
# tools are off (it includes headers only the tool targets generate).
# LLAMA_BUILD_MTMD=ON: the multimodal library (bench-gemma3-llamacpp) lives
# under tools/, so it is built and installed standalone.
# LLAMA_BUILD_NUMBER: a shallow clone has no tags for llama.cpp to derive it
# from, and the benches check it.
#------------------------------------------------------------------------------
echo "[2/3] building (this is the slow part — a few minutes)..."
# shellcheck disable=SC2016  # $ORIGIN is for the linker, not the shell
cmake -S "$SRC" -B "$SRC/build" "${GENERATOR[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS="$SHARED" \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_MTMD=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_NUMBER="$BUILD_NUMBER" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' > "$SRC/configure.log" 2>&1 || {
        echo "configure failed — tail of $SRC/configure.log:" >&2
        tail -20 "$SRC/configure.log" >&2
        exit 1
    }

cmake --build "$SRC/build" -j "$JOBS" > "$SRC/build.log" 2>&1 || {
    echo "build failed — tail of $SRC/build.log:" >&2
    tail -30 "$SRC/build.log" >&2
    exit 1
}

# The old install is replaced only now that the new build has succeeded, so a
# failure above leaves it in place. Signals are held off for the few seconds
# between removing it and finishing the copy, so a Ctrl-C cannot leave a
# half-installed prefix (and the cleanup trap above is retired with it).
echo "[3/3] installing..."
trap '' INT TERM
rm -rf "$PREFIX"
cmake --install "$SRC/build" > "$SRC/install.log" 2>&1 || {
    echo "install failed — tail of $SRC/install.log:" >&2
    tail -20 "$SRC/install.log" >&2
    exit 1
}
trap - INT TERM

[ -f "$CONFIG" ] || { echo "install finished but $CONFIG is missing" >&2; exit 1; }
register

keep_converter

if [ $KEEP_SRC -eq 0 ]; then
    rm -rf "$SRC"
else
    echo "      source kept at $SRC"
fi

#------------------------------------------------------------------------------
# Done
#------------------------------------------------------------------------------
echo
echo "installed llama.cpp $TAG ($(du -sh "$PREFIX" | cut -f1)) at:"
echo "  $PREFIX"
echo "registered for find_package(llama) in $REG_DIR"
echo "GGUF converter at $CONVERT (see benchmarks/README.md, GGUF Conversion)"
echo
echo "Next:"
echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON"
echo "  cmake --build build"
echo "  build/bin/bench-gpt2-vai"
