#!/usr/bin/env bash
# GitHub Ubuntu mirrorlist can strand apt on Azure even after fetching an
# index from the official fallback. Keep official URLs and bound network waits.
set -euo pipefail
mirror_file=${CRISPASR_CI_APT_MIRRORS_FILE:-/etc/apt/apt-mirrors.txt}
if [[ -f "$mirror_file" ]]; then
    sudo sed -i 's|http://azure.archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' "$mirror_file"
fi
exec sudo apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 \
    -o Acquire::https::Timeout=30 "$@"
