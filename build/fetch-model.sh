#!/usr/bin/env bash
# Puts onnx.tar.gz, the model archive of the Python client of Chroma, in artifacts/model, checked against its SHA-256 hash, for the
# tests of the download and the reference of Python. An archive already there with the hash is kept.
set -euo pipefail

url=https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz
hash=913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3
dir="$(cd "$(dirname "$0")/.." && pwd)/artifacts/model"
archive="$dir/onnx.tar.gz"

sha256() {
  if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p "$dir"
if [ -f "$archive" ] && [ "$(sha256 "$archive")" = "$hash" ]; then
  echo "$archive: already there, hash ok"
  exit 0
fi
curl -fsSL --retry 3 -o "$archive.tmp" "$url"
actual=$(sha256 "$archive.tmp")
if [ "$actual" != "$hash" ]; then
  rm -f "$archive.tmp"
  echo "$url has the SHA-256 hash $actual, not $hash" >&2
  exit 1
fi
mv -f "$archive.tmp" "$archive"
echo "$archive: downloaded, hash ok"
