Preview of ChromaDotNet.Embeddings VERSION: the embedding functions of Chroma for .NET, computed locally. The default one:
all-MiniLM-L6-v2 in ONNX, with the same vectors as the Python client. The hosted ones of Chroma Cloud: chroma-cloud-qwen
(Qwen3-Embedding-0.6B) and chroma-cloud-splade (SPLADE), without an account or an API key. Independent project. Not affiliated with, endorsed by or sponsored by Chroma Inc. CHROMA is a registered trademark of Chroma Inc., used here only to describe compatibility.

The package is on NuGet: [ChromaDotNet.Embeddings](https://www.nuget.org/packages/ChromaDotNet.Embeddings/VERSION). It runs on .NET 8
and later, and on .NET Framework 4.6.2 and later on Windows, in 64-bit processes.

The files of this release are the packages that the CI built, and their build provenance. To check a package:

```
gh attestation verify ChromaDotNet.Embeddings.VERSION.nupkg --repo ChromaDotNet/ChromaDB.Embeddings
```
