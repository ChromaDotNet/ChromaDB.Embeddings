"""The Python client of Chroma with CHROMA_EMBED_URL pointing at ChromaDotNet.Embeddings.Server: chroma-cloud-qwen and
chroma-cloud-splade of chromadb compute through the service, and the vectors are those Chroma Cloud recorded on 25 October 2026.
Qwen goes through a collection: of a local Chroma (EphemeralClient), or of the server at the URL given after the file, like a
VecturaDB: add, get with the embeddings, query.
It fails beyond the tolerances of the tests of ChromaDotNet.Embeddings: 1 - cosine 1e-5 for Qwen, values 0.06 apart for SPLADE.
    CHROMA_EMBED_URL=http://127.0.0.1:8080 CHROMA_API_KEY=... python -I build/server-clients/python_client_check.py         tests/ChromaDotNet.Embeddings.Tests/TestData/hosted/cloud-v2.json [http://127.0.0.1:8000]"""
import json
import sys

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import ChromaCloudQwenEmbeddingFunction, ChromaCloudSpladeEmbeddingFunction
from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import ChromaCloudQwenEmbeddingModel

cloud = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"chromadb {chromadb.__version__}")

qwen = ChromaCloudQwenEmbeddingFunction(model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B, task="nl_to_code")
if len(sys.argv) > 2:
    from urllib.parse import urlparse
    url = urlparse(sys.argv[2])
    client = chromadb.HttpClient(host=url.hostname, port=url.port)
    print(f"  server of the collection: {sys.argv[2]}, {client.get_version()}")
    try:
        client.delete_collection("embed-server-check")
    except Exception:
        pass
else:
    client = chromadb.EphemeralClient()
collection = client.create_collection("embed-server-check", embedding_function=qwen, configuration={"hnsw": {"space": "cosine"}})
documents = cloud["qwen-documents"]["request"]["texts"]
collection.add(ids=[str(i) for i in range(len(documents))], documents=documents)
stored = collection.get(ids=[str(i) for i in range(len(documents))], include=["embeddings"])["embeddings"]
a, b = np.asarray(stored, dtype=np.float64), np.asarray(cloud["qwen-documents"]["vectors"], dtype=np.float64)
cosines = 1 - (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
print(f"  qwen documents in the collection: {len(a)}, 1 - cosine at most {cosines.max():.2e}, max abs diff {np.abs(a - b).max():.2e}")
assert len(a) == len(documents) and cosines.max() <= 1e-5

queries = cloud["qwen-queries"]["request"]["texts"]
local = np.asarray(qwen.embed_query(queries), dtype=np.float64)
remote = np.asarray(cloud["qwen-queries"]["vectors"], dtype=np.float64)
cosines = 1 - (local * remote).sum(1) / np.linalg.norm(local, axis=1) / np.linalg.norm(remote, axis=1)
print(f"  qwen queries with the instruction: 1 - cosine at most {cosines.max():.2e}")
assert cosines.max() <= 1e-5
results = collection.query(query_texts=queries, n_results=5)["ids"]
expected = [[str(i) for i in np.argsort(-(b @ q / np.linalg.norm(b, axis=1) / np.linalg.norm(q)))[:5]] for q in remote]
same = sum(r == e for r, e in zip(results, expected))
print(f"  queries with the first five documents of Chroma Cloud: {same} of {len(queries)}")
assert same == len(queries)
print(f"  embedding function of the collection: {client.get_collection('embed-server-check').configuration.get('embedding_function').name()}")

splade = ChromaCloudSpladeEmbeddingFunction()
worst = 0.0
for name in ("splade-documents", "splade-queries"):
    vectors = splade(cloud[name]["request"]["texts"])
    for mine, theirs in zip(vectors, cloud[name]["vectors"]):
        remote = dict(zip(theirs["indices"], theirs["values"]))
        worst = max([worst] + [abs(v - remote.get(i, 0.0)) for i, v in zip(mine.indices, mine.values)])
print(f"  splade: values up to {worst:.2e} from those of Chroma Cloud")
assert worst <= 0.06
tokens = ChromaCloudSpladeEmbeddingFunction(include_tokens=True)(["A sour lemon"])[0]
print(f"  splade with include_tokens: {len(tokens.indices)} terms, labels {tokens.labels[:4] if tokens.labels else None}")
assert tokens.labels is not None and len(tokens.labels) == len(tokens.indices) and "lemon" in tokens.labels
