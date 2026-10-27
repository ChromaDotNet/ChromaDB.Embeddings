"""The reference of Python for the tests of the hosted models: the same ONNX files with onnxruntime 1.24.4, the version of the package,
on the texts of tests/ChromaDotNet.Embeddings.Tests/TestData/hosted/cloud-v2.json.

Usage: python -I build/python-hosted-reference.py <Qwen3 fp32 folder> <Qwen3 fp16 folder> <SPLADE folder> [--write]

Each folder has the files of the model in the layout of its repository on Hugging Face: tokenizer.json and onnx/. As the package does:
Qwen3-Embedding-0.6B one text at a time, input_ids, an attention mask of ones, the positions 0 to n-1 and the past_key_values empty, the
last token of last_hidden_state normalized to length 1; the queries as "Instruct: {instruction}\nQuery:{text}". SPLADE one text at a
time, truncated to 512 tokens, the maximum over the tokens of log(1 + ReLU(logit)), the values above zero. The sessions run as those of
the package: sequential, one thread between operators, half of the processors within an operator.

Without --write, the script computes the token ids and the vectors again and compares them with TestData/hosted/python-onnx.json: the
same ids, the same indices, and values within a tolerance for another processor. With --write, it writes the file.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tokenizers

DATA = Path(__file__).resolve().parent.parent / "tests" / "ChromaDotNet.Embeddings.Tests" / "TestData" / "hosted"
OUTPUT = DATA / "python-onnx.json"
# Another processor runs other kernels: the largest differences measured between two machines are far below these.
DENSE_TOLERANCE = 1e-4
SPARSE_TOLERANCE = 1e-3

qwen32, qwen16, splade_dir = (Path(a) for a in sys.argv[1:4])
write = "--write" in sys.argv[4:]
cloud = json.loads((DATA / "cloud-v2.json").read_text(encoding="utf-8"))
instruction = cloud["qwen-queries"]["request"]["instructions"]
texts = {
    "documents": cloud["qwen-documents"]["request"]["texts"],
    "queries": cloud["qwen-queries"]["request"]["texts"],
}
assert texts["documents"] == cloud["splade-documents"]["request"]["texts"]
assert texts["queries"] == cloud["splade-queries"]["request"]["texts"]

options = ort.SessionOptions()
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.inter_op_num_threads = 1
options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)


def float32s(values):
    """float32 values with 9 significant digits, which give back the same float32: half the size of the repr of a float64."""
    return [float(f"{x:.9g}") for x in np.asarray(values, dtype=np.float32).tolist()]


def qwen(folder, model_file):
    tokenizer = tokenizers.Tokenizer.from_file(str(folder / "tokenizer.json"))
    session = ort.InferenceSession(str(folder / "onnx" / model_file), options, providers=["CPUExecutionProvider"])
    past = {i.name: i for i in session.get_inputs() if i.name.startswith("past_key_values.")}
    out = {}
    for kind, items in texts.items():
        ids_list, vectors = [], []
        for text in items:
            ids = tokenizer.encode(f"Instruct: {instruction}\nQuery:{text}" if kind == "queries" else text).ids
            n = len(ids)
            feed = {
                "input_ids": np.asarray([ids], dtype=np.int64),
                "attention_mask": np.ones((1, n), dtype=np.int64),
                "position_ids": np.arange(n, dtype=np.int64)[None, :],
            }
            for name, i in past.items():
                feed[name] = np.zeros([1, i.shape[1], 0, i.shape[3]], dtype=np.float16 if "float16" in i.type else np.float32)
            last = np.asarray(session.run(["last_hidden_state"], feed)[0][0, -1], dtype=np.float64)
            ids_list.append(ids)
            vectors.append(float32s(last / np.linalg.norm(last)))
        out[kind] = {"ids": ids_list, "vectors": vectors}
    return out


def splade(folder):
    tokenizer = tokenizers.Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(512)
    session = ort.InferenceSession(str(folder / "onnx" / "model.onnx"), options, providers=["CPUExecutionProvider"])
    out = {}
    for kind, items in texts.items():
        rows = {"ids": [], "indices": [], "values": []}
        for text in items:
            ids = tokenizer.encode(text).ids
            n = len(ids)
            logits = session.run(None, {"input_ids": np.asarray([ids], dtype=np.int64), "input_mask": np.ones((1, n), dtype=np.int64),
                                        "segment_ids": np.zeros((1, n), dtype=np.int64)})[0][0]
            v = np.log1p(np.maximum(logits.astype(np.float64), 0)).max(axis=0).astype(np.float32)
            nz = np.nonzero(v > 0)[0]
            rows["ids"].append(ids)
            rows["indices"].append(nz.tolist())
            rows["values"].append(float32s(v[nz]))
        out[kind] = rows
    return out


result = {
    "versions": {"onnxruntime": ort.__version__, "tokenizers": tokenizers.__version__, "numpy": np.__version__},
    "threads": options.intra_op_num_threads,
    "qwen-fp32": qwen(qwen32, "model.onnx"),
    "qwen-fp16": qwen(qwen16, "model_fp16.onnx"),
    "splade": splade(splade_dir),
}
print(f"onnxruntime {ort.__version__}, tokenizers {tokenizers.__version__}, {options.intra_op_num_threads} threads")
if write:
    OUTPUT.write_text(json.dumps(result), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    sys.exit(0)

reference = json.loads(OUTPUT.read_text(encoding="utf-8"))
ok = True
for variant in ("qwen-fp32", "qwen-fp16"):
    for kind in texts:
        got, want = result[variant][kind], reference[variant][kind]
        same_ids = got["ids"] == want["ids"]
        diff = float(np.abs(np.asarray(got["vectors"]) - np.asarray(want["vectors"])).max())
        good = same_ids and diff <= DENSE_TOLERANCE
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {variant} {kind}: ids identical {same_ids}, max abs diff {diff:.3e} (tolerance {DENSE_TOLERANCE:g})")
for kind in texts:
    got, want = result["splade"][kind], reference["splade"][kind]
    same_ids = got["ids"] == want["ids"]
    # A value near zero may be above zero on one processor only: the indices of either side, a missing one as zero.
    diff = 0.0
    for gi, gv, wi, wv in zip(got["indices"], got["values"], want["indices"], want["values"]):
        g, w = dict(zip(gi, gv)), dict(zip(wi, wv))
        diff = max([diff] + [abs(g.get(i, 0.0) - w.get(i, 0.0)) for i in set(g) | set(w)])
    good = same_ids and diff <= SPARSE_TOLERANCE
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} splade {kind}: ids identical {same_ids}, max abs diff {diff:.3e} (tolerance {SPARSE_TOLERANCE:g})")
sys.exit(0 if ok else 1)
