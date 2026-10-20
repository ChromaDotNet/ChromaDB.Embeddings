"""The reference of Python for the tests: the default embedding function of chromadb (ONNXMiniLM_L6_V2) on the texts of
tests/ChromaDotNet.Embeddings.Tests/TestData.

Usage: python -I build/python-reference.py <model folder> [--write]

The model folder is the one ONNXMiniLM_L6_V2 uses, with onnx.tar.gz or onnx/ in it; Python downloads the archive there when it is
missing. Without --write, the script computes the token ids and the vectors of corpus.json and of the texts of python-special.json,
and compares them with python.json and python-special.json: the same ids, and vectors within the tolerance of the tests. With
--write, it writes the two files again.
"""
import json
import sys
from pathlib import Path

import numpy as np

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

TOLERANCE = 1e-6
DATA = Path(__file__).resolve().parent.parent / "tests" / "ChromaDotNet.Embeddings.Tests" / "TestData"

ONNXMiniLM_L6_V2.DOWNLOAD_PATH = Path(sys.argv[1])
write = "--write" in sys.argv[2:]
ef = ONNXMiniLM_L6_V2()
ef._download_model_if_not_exists()
versions = {"chromadb": chromadb.__version__, "onnxruntime": ef.ort.__version__}
print(f"chromadb {versions['chromadb']}, onnxruntime {versions['onnxruntime']}")


def compute(texts):
    ids = [e.ids for e in [ef.tokenizer.encode(t) for t in texts]]
    vectors = np.array(ef(texts), dtype=np.float32)
    return ids, vectors


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8", newline="\n")
    print(f"wrote {path}")


def compare(name, ids, vectors, expected_ids, expected_vectors):
    same_ids = sum(a == b for a, b in zip(ids, expected_ids))
    diff = float(np.abs(vectors - np.array(expected_vectors, dtype=np.float32)).max())
    ok = len(ids) == len(expected_ids) and same_ids == len(ids) and diff <= TOLERANCE
    print(f"{'ok  ' if ok else 'FAIL'} {name}: ids identical {same_ids}/{len(expected_ids)}, max abs diff {diff:.3e} (tolerance {TOLERANCE:g})")
    return ok


corpus = [item["text"] for item in json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))]
special = json.loads((DATA / "python-special.json").read_text(encoding="utf-8"))
corpus_ids, corpus_vectors = compute(corpus)
special_ids, special_vectors = compute(special["texts"])

if write:
    write_json(DATA / "python.json", {"versions": versions, "ids": corpus_ids, "batch32": corpus_vectors.tolist()})
    write_json(DATA / "python-special.json", {"versions": versions, "texts": special["texts"], "ids": special_ids, "vectors": special_vectors.tolist()})
    sys.exit(0)

reference = json.loads((DATA / "python.json").read_text(encoding="utf-8"))
ok = compare("corpus.json", corpus_ids, corpus_vectors, reference["ids"], reference["batch32"])
ok = compare("python-special.json", special_ids, special_vectors, special["ids"], special["vectors"]) and ok
sys.exit(0 if ok else 1)
