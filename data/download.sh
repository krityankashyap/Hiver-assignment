#!/usr/bin/env bash
#
# download.sh — fetch the raw dataset into this folder.
#
# Dataset: "Customer Support on Twitter" (twcs.csv)
# Source:  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
#
# The CSV itself is NOT committed to git (see ../.gitignore). Run this script
# once to populate data/ on your machine.
#
# Requires the Kaggle CLI + API token:
#   pip install kaggle
#   # place your kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
#
# Usage:
#   bash data/download.sh
#
set -euo pipefail

# Resolve the directory this script lives in, so it works from any CWD.
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${DATA_DIR}/archive"
SLUG="thoughtvector/customer-support-on-twitter"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "error: kaggle CLI not found. Install with:  pip install kaggle" >&2
  echo "Then add your API token at ~/.kaggle/kaggle.json" >&2
  exit 1
fi

mkdir -p "${ARCHIVE_DIR}"

echo "Downloading ${SLUG} into ${ARCHIVE_DIR} ..."
kaggle datasets download -d "${SLUG}" -p "${ARCHIVE_DIR}" --unzip

echo "Done. Expected file: ${ARCHIVE_DIR}/twcs/twcs.csv"
