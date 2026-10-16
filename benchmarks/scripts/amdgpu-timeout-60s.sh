#!/usr/bin/env bash
set -euo pipefail

GRUB_FILE="/etc/default/grub"
TIMEOUT="2000,60000,2000,2000"

echo "[1/4] 기존 GRUB 설정 백업..."
sudo cp -a "$GRUB_FILE" "${GRUB_FILE}.bak.$(date +%Y%m%d_%H%M%S)"

echo "[2/4] 기존 amdgpu.lockup_timeout 제거..."
sudo sed -i -E \
  '/^GRUB_CMDLINE_LINUX_DEFAULT=/ s/ ?amdgpu\.lockup_timeout=[^" ]+//g' \
  "$GRUB_FILE"

echo "[3/4] AMDGPU Compute timeout을 60초로 설정..."
sudo sed -i -E \
  "s/^(GRUB_CMDLINE_LINUX_DEFAULT=\"[^\"]*)\"/\1 amdgpu.lockup_timeout=${TIMEOUT}\"/" \
  "$GRUB_FILE"

echo
echo "적용될 설정:"
grep '^GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB_FILE"

echo
echo "[4/4] GRUB 업데이트..."
sudo update-grub

echo
echo "설정 완료. 지금 재부팅합니다."
sudo reboot