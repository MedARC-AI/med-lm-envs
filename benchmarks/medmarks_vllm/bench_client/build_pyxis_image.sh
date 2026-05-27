#!/usr/bin/env bash
set -euo pipefail

BASE_SQSH="${BASE_SQSH:-/path/to/pyxis-images/vllm/latest.sqsh}"
IMAGE_DIR="${IMAGE_DIR:-/path/to/pyxis-images/vllm-bench-client}"
CONTAINER_NAME="${CONTAINER_NAME:-medmarks-vllm-bench-client}"

mkdir -p "${IMAGE_DIR}"

enroot_root="${IMAGE_DIR}/.enroot"
mkdir -p "${enroot_root}/cache" "${enroot_root}/data" "${enroot_root}/runtime" "${enroot_root}/temp" "${enroot_root}/config"

export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${enroot_root}/cache}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-${enroot_root}/data}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-${enroot_root}/runtime}"
export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-${enroot_root}/temp}"
export ENROOT_CONFIG_PATH="${ENROOT_CONFIG_PATH:-${enroot_root}/config}"

if [[ ! -f "${BASE_SQSH}" ]]; then
  echo "BASE_SQSH does not exist: ${BASE_SQSH}" >&2
  exit 1
fi

enroot remove -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
enroot create -f -n "${CONTAINER_NAME}" "${BASE_SQSH}"

rc_script="${IMAGE_DIR}/.exec-args.rc"
cat > "${rc_script}" <<'EOF'
#!/bin/sh
exec "$@"
EOF
chmod +x "${rc_script}"

enroot start --root --rw \
  -e NVIDIA_VISIBLE_DEVICES=void \
  --rc "${rc_script}" \
  "${CONTAINER_NAME}" \
  python3 -m pip install --no-cache-dir pandas

final_image="${IMAGE_DIR}/latest.sqsh"
tmp_image="${IMAGE_DIR}/.latest.${$}.sqsh"

enroot export -f -o "${tmp_image}" "${CONTAINER_NAME}"
mv -f "${tmp_image}" "${final_image}"

enroot remove -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "${final_image}"
