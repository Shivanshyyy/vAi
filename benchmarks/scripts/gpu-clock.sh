#!/bin/sh
# Pin / restore GPU clocks for benchmarking. Picks the vendor by itself.
#
#   gpu-clock.sh status        current clocks and what can be locked (default)
#   gpu-clock.sh lock [MHz]    NVIDIA: lock the graphics clock at <MHz> and the
#                              memory clock at its highest level (MHz required)
#                              AMD:    pin every DPM level to its highest state
#                                      (power_dpm_force_performance_level=high)
#   gpu-clock.sh reset         restore the default boost / auto behaviour
#
# A lock is an upper bound: power and thermal limits still override it, so
# on a laptop or APU pin a clock the GPU sustains under load, not the boost
# peak (watch `nvidia-smi dmon -s c` or pp_dpm_sclk during a run). Record the
# value in the benchmark README next to the driver version.
#
# GPU=<index> selects the device on multi-GPU machines (NVIDIA, default 0).
# GPU_VENDOR=nvidia|amdgpu overrides the detection — needed on hybrid laptops
# (AMD iGPU + NVIDIA dGPU), where the amdgpu card is found first.
set -eu

cmd="${1:-status}"
gpu="${GPU:-0}"

# ----------------------------------------------------------------------------
# Vendor detection: an amdgpu card under /sys/class/drm, else nvidia-smi.
# Cards are found by driver, not number — indices move across boots.
# ----------------------------------------------------------------------------
amd_dev=""
for c in /sys/class/drm/card*; do
    [ -e "$c/device/driver" ] || continue
    if [ "$(basename "$(readlink -f "$c/device/driver")")" = amdgpu ]; then
        amd_dev="$c/device"
        break
    fi
done

vendor="${GPU_VENDOR:-}"
if [ -z "$vendor" ]; then
    if [ -n "$amd_dev" ]; then vendor=amdgpu
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then vendor=nvidia
    else echo "error: no amdgpu device under /sys/class/drm and no working nvidia-smi" >&2; exit 1
    fi
fi

need_root() {
    if [ "$(id -u)" -ne 0 ]; then
        exec sudo GPU="$gpu" GPU_VENDOR="$vendor" "$0" "$@"
    fi
}

# ----------------------------------------------------------------------------
# NVIDIA — nvidia-smi clock locks
# ----------------------------------------------------------------------------
nv_query() { nvidia-smi -i "$gpu" --query-gpu="$1" --format=csv,noheader,nounits; }

nvidia_status() {
    echo "nvidia GPU $gpu: $(nv_query name) (driver $(nv_query driver_version))"
    echo "  graphics: $(nv_query clocks.gr) MHz now, lockable $(nvidia-smi -i "$gpu" --query-supported-clocks=gr --format=csv,noheader,nounits | sort -n | sed -n '1p;$p' | paste -sd- -) MHz"
    echo "  memory:   $(nv_query clocks.mem) MHz now, levels $(nvidia-smi -i "$gpu" --query-supported-clocks=mem --format=csv,noheader,nounits | sort -un | paste -sd' ' -) MHz"
    echo "  (clocks only reach a lock under load; idle reads stay low)"
}

nvidia_lock() {
    mhz="${1:-}"
    case "$mhz" in
        ''|*[!0-9]*|0*) echo "usage: $0 lock <MHz>   (NVIDIA needs a clock; see status for the range)" >&2; exit 2 ;;
    esac
    need_root lock "$mhz"
    mem="$(nvidia-smi -i "$gpu" --query-supported-clocks=mem --format=csv,noheader,nounits | sort -un | tail -1)"
    # Persistence mode keeps the driver (and with it the lock) loaded between
    # bench processes on a headless box. It is left on by reset: harmless, and
    # the previous state is unknown.
    nvidia-smi -i "$gpu" -pm 1 >/dev/null
    nvidia-smi -i "$gpu" -lgc "$mhz,$mhz" >/dev/null
    nvidia-smi -i "$gpu" -lmc "$mem,$mem" >/dev/null
    echo "nvidia GPU $gpu: locked graphics $mhz MHz, memory $mem MHz"
}

nvidia_reset() {
    need_root reset
    nvidia-smi -i "$gpu" -rgc >/dev/null
    nvidia-smi -i "$gpu" -rmc >/dev/null
    echo "nvidia GPU $gpu: clocks reset to default (boost)"
}

# ----------------------------------------------------------------------------
# AMD — amdgpu DPM performance level in sysfs
# ----------------------------------------------------------------------------
amdgpu_level() {
    amd_level="$amd_dev/power_dpm_force_performance_level"
    [ -f "$amd_level" ] || { echo "error: $amd_level not found" >&2; exit 1; }
}

amdgpu_status() {
    amdgpu_level
    echo "amdgpu $(basename "$(dirname "$amd_dev")"): level $(cat "$amd_level")"
    # pp_dpm_* list every DPM state, the active one marked with '*'
    for f in sclk mclk; do
        [ -f "$amd_dev/pp_dpm_$f" ] || continue
        printf '  %s: %s\n' "$f" "$(tr '\n' ' ' < "$amd_dev/pp_dpm_$f")"
    done
}

amdgpu_lock() {
    amdgpu_level
    [ -z "${1:-}" ] || echo "note: amdgpu pins DPM to its highest state; the MHz argument is ignored" >&2
    need_root lock
    echo high > "$amd_level"
    echo "amdgpu $(basename "$(dirname "$amd_dev")"): level $(cat "$amd_level")"
}

amdgpu_reset() {
    amdgpu_level
    need_root reset
    echo auto > "$amd_level"
    echo "amdgpu $(basename "$(dirname "$amd_dev")"): level $(cat "$amd_level")"
}

# ----------------------------------------------------------------------------
case "$cmd" in
    status) "${vendor}_status" ;;
    lock)   "${vendor}_lock" "${2:-}" ;;
    reset)  "${vendor}_reset" ;;
    *)      echo "usage: $0 [status|lock [MHz]|reset]" >&2; exit 2 ;;
esac
