# ChromaDotNet.Embeddings

The default embedding function of Chroma, for .NET.
It computes the vectors that the Python client of Chroma computes when a collection has no embedding function of its own.
The model is all-MiniLM-L6-v2 in ONNX, with 384 dimensions.
The generator is an `IEmbeddingGenerator<string, Embedding<float>>` of Microsoft.Extensions.AI.

Community project, not affiliated with Chroma.

> Preview: the API and the package can still change.

```
dotnet add package ChromaDotNet.Embeddings --prerelease
```

## With ChromaDotNet.Client

[ChromaDotNet.Client](https://www.nuget.org/packages/ChromaDotNet.Client) 2.17.0 and later take the generator.
The client of a collection computes the embeddings of the documents and of the query texts.
A new collection declares the embedding function `default`, as the Python client does.
So Python reads the collection with its default embedding function, and the other way round.

```csharp
using ChromaDB.Client;
using ChromaDB.Client.Models;
using ChromaDotNet.Embeddings;

using var embeddings = await ChromaDefaultEmbeddingGenerator.CreateAsync();
using var client = new ChromaClient("http://localhost:8000/api/v2/");
var collection = await client.GetOrCreateCollectionAsync(new ChromaCollectionDefinition("fruits") { EmbeddingGenerator = embeddings });
var fruits = client.GetCollectionClient(collection).WithEmbeddingGenerator(embeddings);

await fruits.AddAsync(["apple", "lemon", "banana"], documents: ["A crunchy apple", "A sour lemon", "A sweet banana"]);
var results = await fruits.QueryAsync("something sour", nResults: 2, include: ChromaQueryInclude.Documents | ChromaQueryInclude.Distances);
```

## Without the client

```csharp
using ChromaDotNet.Embeddings;
using Microsoft.Extensions.AI;

using var embeddings = await ChromaDefaultEmbeddingGenerator.CreateAsync();
var vectors = await embeddings.GenerateAsync(["A crunchy apple", "A sour lemon"]);
ReadOnlyMemory<float> query = await embeddings.GenerateVectorAsync("something sour");
```

Each vector has 384 dimensions and an L2 norm of 1.
With an earlier version of ChromaDotNet.Client, pass these vectors to `AddAsync` and `QueryAsync`.

## Platforms

| | .NET 8 and later | .NET Framework 4.6.2 and later |
|---|---|---|
| Windows x64 | yes | yes, in a 64-bit process |
| Windows arm64 | yes | yes, in a 64-bit process |
| Linux x64 and arm64 | yes | |
| macOS arm64 | yes | |
| macOS x64 (Intel Macs) | no | |
| 32-bit processes | no | no |

Intel Macs are not supported.
ONNX Runtime has no osx-x64 library from version 1.24 on, and this package uses 1.24.

32-bit processes are not supported.
ONNX Runtime has no win-x86 library from version 1.23 on.

On .NET Framework, set PlatformTarget to x64 or ARM64, or keep AnyCPU without Prefer32Bit.
An AnyCPU application runs as a 64-bit process on 64-bit Windows.
The build of a 32-bit application fails with a message that says so.
.NET Framework does not load native libraries from `runtimes/`.
So the package copies `hf_tokenizers.dll` and `onnxruntime.dll` next to the application at the build.

On .NET 8, the library is compatible with trimming and NativeAOT.

## The model and its cache

The model is not in the package.
The first `CreateAsync` downloads the archive that Python downloads:
`https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz`, 83 MB.
It checks the SHA-256 hash of the archive, and does not keep a file with another hash.
It extracts the six files of `onnx/`, by their exact names.

The cache is the folder of Python: `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2`.
On Windows it is `%USERPROFILE%\.cache\chroma\onnx_models\all-MiniLM-L6-v2`.
A model that Python downloaded is used as it is.
Python uses a model downloaded here.

As in Python, the six files of `onnx/` are used when they are all there.
Otherwise `onnx.tar.gz` is used when its hash matches, or downloaded again.
Two processes that start at the same time download it once, through a lock file next to the archive.
`tokenizer.json` truncates and pads to 128 tokens; Python changes both to 256.
So a copy without them, `tokenizer.untruncated.json`, goes next to it.

Without the network and without the model in the cache, `CreateAsync` throws an `InvalidOperationException`.
Its message has the address it tried, and the options to work offline.

| `ChromaDefaultEmbeddingOptions` | Default | |
|---|---|---|
| `CacheDirectory` | the folder of Python | Where the archive and `onnx/` go. |
| `ModelDirectory` | none | An `onnx/` folder already extracted: no download, no cache. |
| `DownloadUri` | the address of Python | A mirror of `onnx.tar.gz`, with the same hash. |
| `IntraOpNumThreads` | half of the processors, at least 1 | The threads of ONNX Runtime for a batch. |
| `Logger` | none | The download, the extraction and the load. |

## What it computes

It does what `ONNXMiniLM_L6_V2` of chromadb 1.5.9 does:

- the tokenizer of `tokenizer.json`, with `[CLS]` and `[SEP]`;
- truncation and padding to 256 tokens;
- `token_type_ids` of zeros;
- the mean of `last_hidden_state` over the tokens of the attention mask;
- L2 normalization, in float32;
- batches of 32. A batch gives the same vectors as each text alone.

The tests compare 135 texts with Python: empty texts, accents, emoji, CJK and other scripts, symbols, control characters, texts of
254 to 1000 tokens, and short documents.
The token ids are the same for all of them.
The largest difference of a component is 1.9e-7.
The five nearest neighbours of each text are the same.
The reference of Python is in `tests/ChromaDotNet.Embeddings.Tests/TestData`.

The constants of `ChromaDefaultEmbeddingGenerator` are those of Python for a collection created without an embedding function:

- `EmbeddingFunctionName` is `default`, the name in the configuration of the collection;
- `DefaultSpace` is `l2`;
- `Dimensions` is 384, `MaxTokens` is 256.

`GetService(typeof(EmbeddingGeneratorMetadata))` gives the provider `chroma` and the model `all-MiniLM-L6-v2`.

The computation runs on the CPU, in the calling thread.
Concurrent calls run one after the other.

## Built on

- [Tokenizers.DotNet](https://www.nuget.org/packages/Tokenizers.DotNet), by Taein Kim: the tokenizers library of Hugging Face,
  the one of Python. [GitHub](https://github.com/sappho192/Tokenizers.DotNet)
- [ElBruno.LocalEmbeddings](https://www.nuget.org/packages/ElBruno.LocalEmbeddings), by Bruno Capuano, on .NET 8 and later.
  Its `OnnxEmbeddingModel` runs the model, with the mean pooling and the L2 normalization.
  [GitHub](https://github.com/elbruno/elbruno.localembeddings)
- [ONNX Runtime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime), directly on .NET Framework, which
  ElBruno.LocalEmbeddings does not target. [GitHub](https://github.com/microsoft/onnxruntime)

## Licenses

The code of this repository is under the MIT License.
The model is downloaded, not distributed here: all-MiniLM-L6-v2 of sentence-transformers is under the Apache License 2.0.
[THIRD-PARTY-NOTICES.md](https://github.com/ChromaDotNet/ChromaDB.Embeddings/blob/main/THIRD-PARTY-NOTICES.md) lists the model and the packages that NuGet installs, with their licenses.

## Build and tests

```
bash build/fetch-model.sh
dotnet test tests/ChromaDotNet.Embeddings.Tests
```

`build/fetch-model.sh` puts `onnx.tar.gz` in `artifacts/model`, checked against its hash.
The tests of the download serve it from a local mirror; without it they are skipped.
The tests run on .NET 8 and on .NET Framework 4.8, and put their caches in `artifacts/test-tmp`.

`build/python-reference.py`, with chromadb 1.5.9, computes the vectors of the test texts in Python again, and compares them with the
reference.
`dotnet run -c Release --project tests/ChromaDotNet.Embeddings.Bench` measures the speed.
