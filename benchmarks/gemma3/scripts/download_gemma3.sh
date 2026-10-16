#!/usr/bin/env bash
set -euo pipefail

echo "=============================================================="
echo " Hugging Face Gemma downloader"
echo " Models:"
echo "   - google/gemma-3-4b-it"
echo "   - google/gemma-3-1b-it"
echo "=============================================================="
echo

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python 명령을 찾을 수 없습니다."
  echo "        먼저 'conda activate vai' 를 실행하세요."
  exit 1
fi

if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "[ERROR] huggingface_hub 패키지를 찾을 수 없습니다."
  echo "        다음 명령으로 설치하세요:"
  echo "        pip install -U huggingface_hub"
  exit 1
fi

echo "[INFO] Python:"
python --version
echo

read -rsp "Hugging Face Access Token 입력: " HF_TOKEN
echo

if [[ -z "${HF_TOKEN}" ]]; then
  echo "[ERROR] Hugging Face 토큰이 비어 있습니다."
  exit 1
fi

read -rp "모델 저장 기본 경로 입력 (예: /workspace/models): " MODEL_BASE_DIR

if [[ -z "${MODEL_BASE_DIR}" ]]; then
  echo "[ERROR] 저장 경로가 비어 있습니다."
  exit 1
fi

MODEL_BASE_DIR="${MODEL_BASE_DIR/#\~/$HOME}"
mkdir -p "${MODEL_BASE_DIR}"

export HF_TOKEN
export MODEL_BASE_DIR

echo
echo "저장 기본 경로: ${MODEL_BASE_DIR}"
echo

python <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

token = os.environ["HF_TOKEN"]
base_dir = Path(os.environ["MODEL_BASE_DIR"]).expanduser().resolve()

models = [
    "google/gemma-3-4b-it",
    "google/gemma-3-1b-it",
]

print("==============================================================")
print(" Hugging Face 인증 확인")
print("==============================================================")

try:
    api = HfApi(token=token)
    user = api.whoami()
    print(f"[OK] Hugging Face 인증 성공: {user.get('name', 'unknown')}")
except Exception as e:
    print(f"[ERROR] Hugging Face 인증 실패: {e}")
    sys.exit(1)

print()
print("==============================================================")
print(" 모델 다운로드 시작")
print("==============================================================")

results = []

for repo_id in models:
    model_name = repo_id.split("/")[-1]
    local_dir = base_dir / model_name

    print()
    print(f"[DOWNLOAD] {repo_id}")
    print(f"           -> {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=str(local_dir),
        )

        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"[OK] {repo_id} 다운로드 완료")
            results.append((repo_id, local_dir, True, None))
        else:
            print(f"[ERROR] {repo_id}: 다운로드 후 파일을 찾을 수 없습니다.")
            results.append((repo_id, local_dir, False, "download directory empty"))

    except Exception as e:
        print(f"[ERROR] {repo_id} 다운로드 실패")
        print(f"        {e}")
        results.append((repo_id, local_dir, False, str(e)))

print()
print("==============================================================")
print(" 다운로드 결과")
print("==============================================================")

failed = False

for repo_id, path, ok, err in results:
    if ok:
        print(f"[SUCCESS] {repo_id}")
        print(f"          {path}")
    else:
        failed = True
        print(f"[FAILED ] {repo_id}")
        print(f"          {path}")
        if err:
            print(f"          {err}")

print()

if failed:
    print("RESULT: FAILED / INCOMPLETE")
    print()
    print("참고:")
    print("- Gemma 모델은 Hugging Face에서 사용 약관 동의/접근 권한 승인이 필요할 수 있습니다.")
    print("- 해당 모델 페이지에서 접근 권한을 승인한 계정의 토큰을 사용하세요.")
    sys.exit(1)

print("RESULT: SUCCESS")
PY

unset HF_TOKEN

echo
echo "완료되었습니다."
