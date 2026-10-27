#!/usr/bin/env bash
# Puts the files of the local models of the hosted embedding functions of Chroma Cloud in artifacts/hosted-models, in the layout of the
# cache of the package: Qwen3-Embedding-0.6B/fp16, Qwen3-Embedding-0.6B/fp32 and Splade_PP_en_v1. Each file comes from Hugging Face at
# the revision of the package and is checked against its SHA-256 hash, as in src/ChromaDotNet.Embeddings/HostedModels.cs: a test checks
# that the two lists are the same. A file already there with its hash is kept.
#   bash build/fetch-hosted-models.sh [qwen-fp16] [qwen-fp32] [splade]    (default: qwen-fp16 splade)
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)/artifacts/hosted-models"

# model, folder in the cache, repository, revision, path in the repository, SHA-256
files=(
  "qwen-fp16 Qwen3-Embedding-0.6B/fp16 onnx-community/Qwen3-Embedding-0.6B-ONNX c25a394dd583836952667c12f008335071b3f43d tokenizer.json def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
  "qwen-fp16 Qwen3-Embedding-0.6B/fp16 onnx-community/Qwen3-Embedding-0.6B-ONNX c25a394dd583836952667c12f008335071b3f43d onnx/model_fp16.onnx f376cf065d912675dd559270e19361367ebc28c938b9811dd61d5834c78d0834"
  "qwen-fp16 Qwen3-Embedding-0.6B/fp16 onnx-community/Qwen3-Embedding-0.6B-ONNX c25a394dd583836952667c12f008335071b3f43d onnx/model_fp16.onnx_data 06fbfded30166e3a33fe5252d7002ed1c370a4ec1bf45aba778cdf6cf7d31439"
  "qwen-fp32 Qwen3-Embedding-0.6B/fp32 onnx-community/Qwen3-Embedding-0.6B-ONNX c25a394dd583836952667c12f008335071b3f43d tokenizer.json def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
  "qwen-fp32 Qwen3-Embedding-0.6B/fp32 onnx-community/Qwen3-Embedding-0.6B-ONNX c25a394dd583836952667c12f008335071b3f43d onnx/model.onnx bf27b2f3f9ef9c32ca337d75b361fa99439deaeaefe82e4701b2dbd8439197cc"
  "qwen-fp32 Qwen3-Embedding-0.6B/fp32 onnx-community/Qwen3-Embedding-0.6B-ONNX c25a394dd583836952667c12f008335071b3f43d onnx/model.onnx_data f0a61604465929a27e68aa6217c8c89ec6186572f0209fdb7711adda48a9b9a9"
  "splade Splade_PP_en_v1 prithivida/Splade_PP_en_v1 762be6a7206e2f299182705972a65e5c46e62be2 tokenizer.json 2fc687b11de0bc1b3d8348f92e3b49ef1089a621506c7661fbf3248fcd54947e"
  "splade Splade_PP_en_v1 prithivida/Splade_PP_en_v1 762be6a7206e2f299182705972a65e5c46e62be2 onnx/model.onnx 0934583a27a031a66b2e847cbc260fbbef29689e969f500436460ef5146a43f2"
)

sha256() {
  if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

models=("$@")
if [ ${#models[@]} -eq 0 ]; then models=(qwen-fp16 splade); fi
for model in "${models[@]}"; do
  case "$model" in qwen-fp16|qwen-fp32|splade) ;; *) echo "unknown model $model: qwen-fp16, qwen-fp32 or splade" >&2; exit 2 ;; esac
  for entry in "${files[@]}"; do
    read -r name folder repository revision path hash <<< "$entry"
    [ "$name" = "$model" ] || continue
    file="$root/$folder/$path"
    if [ -f "$file" ] && [ "$(sha256 "$file")" = "$hash" ]; then
      echo "$model $path: already there, hash ok"
      continue
    fi
    mkdir -p "$(dirname "$file")"
    url="https://huggingface.co/$repository/resolve/$revision/$path"
    curl -fsSL --retry 3 -o "$file.tmp" "$url"
    actual=$(sha256 "$file.tmp")
    if [ "$actual" != "$hash" ]; then
      rm -f "$file.tmp"
      echo "$url has the SHA-256 hash $actual, not $hash" >&2
      exit 1
    fi
    mv -f "$file.tmp" "$file"
    echo "$model $path: downloaded, hash ok"
  done
done
