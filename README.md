# ChromaDotNet.Embeddings

[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/ChromaDotNet/ChromaDB.Embeddings?label=openssf%20scorecard)](https://scorecard.dev/viewer/?uri=github.com/ChromaDotNet/ChromaDB.Embeddings)

The default embedding function of Chroma, for .NET.
It computes the vectors that the Python client of Chroma computes when a collection has no embedding function of its own.
The model is all-MiniLM-L6-v2 in ONNX, with 384 dimensions.
The generator is an `IEmbeddingGenerator<string, Embedding<float>>` of Microsoft.Extensions.AI.
The package also computes the two hosted embedding functions of Chroma Cloud on this machine: `chroma-cloud-qwen` and
`chroma-cloud-splade`.

ChromaDotNet is an independent community project that provides .NET libraries and tools for Chroma. Independent project. Not affiliated with, endorsed by or sponsored by Chroma Inc. CHROMA is a registered trademark of Chroma Inc., used here only to describe compatibility. We use the name "Chroma" only to say which software our projects work with. It does not mean that Chroma Inc. made, approved or supports them.

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

Try it in your browser: [SearchByText](https://github.com/ChromaDotNet/ChromaDB.Local/tree/main/samples/SearchByText) stores and searches texts with this package and ChromaDotNet.Client, on a Chroma started by ChromaDotNet.Local. [Open it in a Codespace](https://codespaces.new/ChromaDotNet/ChromaDB.Local?quickstart=1).

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

## The hosted embedding functions of Chroma Cloud

Chroma Cloud computes two embedding functions on its servers: `chroma-cloud-qwen` and `chroma-cloud-splade`.
This package computes them on this machine, on the CPU, with the same models in ONNX.
It needs no account and no API key.

| | `ChromaQwenEmbeddingGenerator` | `ChromaSpladeEmbeddingGenerator` |
|---|---|---|
| Embedding function | `chroma-cloud-qwen` | `chroma-cloud-splade` |
| Model | Qwen3-Embedding-0.6B, in the ONNX export of onnx-community | SPLADE, prithivida/Splade_PP_en_v1, in the ONNX export of its repository |
| Vectors | dense: 1024 dimensions, length 1 | sparse: a weight for some of the 30522 tokens of the vocabulary |
| Tokens of a text | up to 32768 | up to 512, as in Chroma Cloud; the rest is ignored |
| Download | 1.2 GB in float16, the default; 2.4 GB in float32 | 533 MB |

```csharp
using ChromaDotNet.Embeddings;
using Microsoft.Extensions.AI;

using var qwen = await ChromaQwenEmbeddingGenerator.CreateAsync();
var documents = await qwen.GenerateAsync(["def add(a, b): return a + b", "A sour lemon"]);
ReadOnlyMemory<float> query = await qwen.Queries.GenerateVectorAsync("How do I add two numbers?");

using var splade = await ChromaSpladeEmbeddingGenerator.CreateAsync();
ChromaSparseEmbedding lemon = (await splade.GenerateAsync(["A sour lemon"]))[0];
```

`GenerateAsync` embeds documents, and `Queries` embeds queries: Qwen3-Embedding reads them with different instructions, as Chroma
Cloud does.
The default task is `nl_to_code`, the one the Python client has instructions for.
Its instructions are none for the documents, and "Given a question about coding, retrieval code or passage that can solve user's
question" for the queries.
`Task` and `Instructions` of `ChromaQwenEmbeddingOptions` change them, as `task` and `instructions` do in the Python client.

SPLADE embeds documents and queries in the same way.
`Indices` has the ids of the tokens with a weight, in ascending order, and `Values` has their weights.
With `IncludeTokens`, `Labels` has the token of each index, as `include_tokens` does in the Python client.

With ChromaDotNet.Client, `WithEmbeddingGenerator(qwen)` computes the embeddings of the documents, and
`WithEmbeddingGenerator(qwen.Queries)` computes those of the query texts.
The client declares the embedding function `default` for `ChromaDefaultEmbeddingGenerator` only: with these generators, a new
collection declares none.
The client does not compute SPLADE vectors: give them to it as `ChromaSparseVector`, with the indices as `uint`.

### Their models and their cache

The models are not in the package.
The first `CreateAsync` downloads their files from Hugging Face, at a fixed revision.
It checks the SHA-256 hash of each file, and does not keep a file with another hash.
The next calls check only that each file has its size.
In the cache, a file of another size is downloaded again; in a `ModelDirectory`, it makes `CreateAsync` throw an
`InvalidOperationException`.
The cache is next to the folder of the default model: `~/.cache/chroma/onnx_models/Qwen3-Embedding-0.6B/fp16`, or `fp32`, and
`~/.cache/chroma/onnx_models/Splade_PP_en_v1`.
The files keep the layout of the repository on Hugging Face: `tokenizer.json` and `onnx/`.
Two processes that start at the same time download each file once, through a lock file.
Without the network and without the files in the cache, `CreateAsync` throws an `InvalidOperationException`.
Its message has the missing files with their addresses and hashes, and the options to work offline.

| `ChromaQwenEmbeddingOptions`, `ChromaSpladeEmbeddingOptions` | Default | |
|---|---|---|
| `CacheDirectory` | the folder of the model in `~/.cache/chroma/onnx_models` | Where the files go. |
| `ModelDirectory` | none | A folder with the files, in the layout of the repository: no download, no cache. |
| `DownloadBaseUri` | `https://huggingface.co/` | A mirror of Hugging Face, with the same files. |
| `IntraOpNumThreads` | half of the processors, at least 1 | The threads of ONNX Runtime for a text. |
| `Logger` | none | The download, the load and, for Qwen, the texts truncated. |
| `Variant`, Qwen only | `Float16` | `Float32` downloads and runs the weights in float32. |
| `Task` and `Instructions`, Qwen only | `nl_to_code` and the instructions of the Python client | The instructions of the documents and of the queries. |
| `IncludeTokens`, SPLADE only | false | The token of each index in `Labels`. |

### What they compute

Qwen3-Embedding, as the service of Chroma Cloud:

- the tokenizer of `tokenizer.json`, which ends each text with `<|endoftext|>`;
- a query as `Instruct: {instruction}\nQuery:{text}`, with no space after `Query:`;
- each text alone, without padding;
- a text of more than 32768 tokens truncated, with `<|endoftext|>` at the end; the logger says so;
- the last hidden state of the last token, normalized to length 1.

SPLADE, as the service of Chroma Cloud:

- the tokenizer of `tokenizer.json`, with `[CLS]` and `[SEP]`;
- a text of more than 512 tokens truncated, with `[SEP]` at the end;
- each text alone;
- for each token of the vocabulary, the largest log(1 + ReLU(logit)) over the tokens of the text;
- the weights above zero, by ascending index.

The tests compare the vectors with those that Chroma Cloud returned for the same texts on 25 October 2026.
They are in `tests/ChromaDotNet.Embeddings.Tests/TestData/hosted`.

| | Qwen in float16 | Qwen in float32 | SPLADE |
|---|---|---|---|
| Vectors of Chroma Cloud compared | 69 | 69 | 61 |
| Largest 1 − cosine | 3.8e-6 | 3.0e-6 | |
| Largest difference of a value | 4.8e-4 | 3.7e-4 | 0.045 |
| Terms that only Chroma Cloud has | | | a few, each below 0.016 |
| 12 queries on 34 documents | the same first five, in the same order | the same first five, in the same order | the same first one, and the same first five |
| Largest difference of a score | 7.0e-4 | 5.8e-4 | 0.21, on scores up to 29 |
| The tests allow | 1e-5 for 1 − cosine, 1e-3 for a value | 1e-5 for 1 − cosine, 1e-3 for a value | 0.02 for a term that only Chroma Cloud has, 0.06 for a value |

Other texts can differ more than these.
Each term that SPLADE gives here is also in the vector of Chroma Cloud.
Chroma Cloud gives the SPLADE values in float16, and this package in float32.
Chroma Cloud itself gives values about 0.01 apart to a text alone and to the same text in a batch.
Against onnxruntime 1.24.4 in Python, with the same files, the vectors are the same bit for bit, on the machine that wrote the
reference.
On other processors ONNX Runtime can run other kernels, so the tests allow a difference of 1e-4 for a value of Qwen in float32,
and of 1e-3 for Qwen in float16 and for SPLADE.
On the runners of our CI, Qwen in float16 differed by up to 4.0e-4 from that reference, and by up to 5.4e-4 from Chroma Cloud.
The CI also computes the vectors of Python on each runner, and the tests require those of .NET to be the same bit for bit.
They were, on Windows x64, Linux x64, Linux arm64 and macOS arm64: for Qwen in float16 and SPLADE on each, and for Qwen in float32
on Linux x64, the only runner with its files.

### Their speed

In our tests, these are the times of one text at a time, on the CPU, with the files already downloaded.
The machine is a 13th Gen Intel Core i5-13600KF with 20 logical processors, on Windows 11 and .NET 8, so 10 threads by default.
Before each run, other processes used 19% to 27% of its CPU.
Each value is the median of 5 runs, each in its own process.

| | Qwen in float16 | Qwen in float32 | SPLADE |
|---|---|---|---|
| Load | 2.2 s | 2.0 s | 0.7 s |
| A query of the tests: 28 to 38 tokens with the instruction of Qwen, 7 to 16 for SPLADE | 69 ms | 69 ms | 12 ms |
| A document of 106 tokens for Qwen, 108 for SPLADE | 181 ms | 175 ms | 42 ms |
| A document of 661 tokens for Qwen, truncated to 512 for SPLADE | 1.3 s | 1.2 s | 194 ms |
| A document of 3322 tokens for Qwen, truncated to 512 for SPLADE | 11.7 s | 11.4 s | 195 ms |
| Peak working set of the process | 4.3 GB | 4.8 GB | 0.79 GB |

Qwen is slow to index many long documents: the time of each token grows with the length of the text.
The memory of its attention grows with the square of the tokens of a text: in the ONNX export, the mask of the attention has a value
for each pair of tokens.
In our tests, the peak of Qwen in float16 was 4.3 GB, with texts of up to 3322 tokens.
On a document of about 100 tokens, SPLADE takes about a quarter of the time of Qwen.
SPLADE reads 512 tokens at most, so a longer document takes it no more time.
In float16, Qwen downloads half the bytes of float32, and takes up to 6% more time for a text.
The load took longer in the first of the 5 runs: 2.5 s for Qwen in float32, 1.5 s for SPLADE.
For Qwen in float16, onnxruntime 1.24.4 in Python with the same files takes the same times, within the differences between two runs.
So does a NativeAOT build.

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
The build of a 32-bit application fails with a message that says so, on .NET Framework and on .NET 8 and later.
In a 32-bit process, `CreateAsync` throws a `PlatformNotSupportedException` before any download.
.NET Framework does not load native libraries from `runtimes/`.
So the package copies `hf_tokenizers.dll` and `onnxruntime.dll` next to the application at the build.
The generators of the hosted models run on the same platforms.

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
On these texts, against chromadb 1.5.9 with onnxruntime 1.31.0, the largest difference of a component is 1.9e-7 on x64 and 2.4e-7 on
arm64.
The tests allow 1e-6: other texts can differ more than these.
The five nearest neighbours of each text are the same.
The reference of Python is in `tests/ChromaDotNet.Embeddings.Tests/TestData`.

The constants of `ChromaDefaultEmbeddingGenerator` are those of Python for a collection created without an embedding function:

- `EmbeddingFunctionName` is `default`, the name in the configuration of the collection;
- `DefaultSpace` is `l2`;
- `Dimensions` is 384, `MaxTokens` is 256.

`GetService(typeof(EmbeddingGeneratorMetadata))` gives the provider `chroma` and the model `all-MiniLM-L6-v2`.

The computation runs on the CPU, in the calling thread.
Concurrent calls run one after the other.
ONNX Runtime logs errors only, so Linux without a GPU does not warn at every start that the discovery of the GPU devices failed.
An application that creates `OrtEnv` before keeps its own level.

## Built on

- [Tokenizers.DotNet](https://www.nuget.org/packages/Tokenizers.DotNet), by Taein Kim: the tokenizers library of Hugging Face,
  the one of Python. [GitHub](https://github.com/sappho192/Tokenizers.DotNet)
- [ElBruno.LocalEmbeddings](https://www.nuget.org/packages/ElBruno.LocalEmbeddings), by Bruno Capuano, on .NET 8 and later.
  Its `OnnxEmbeddingModel` runs the model, with the mean pooling and the L2 normalization.
  [GitHub](https://github.com/elbruno/elbruno.localembeddings)
- [ONNX Runtime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime), directly on .NET Framework, which
  ElBruno.LocalEmbeddings does not target, and for the hosted models. [GitHub](https://github.com/microsoft/onnxruntime)

## Licenses

The code of this repository is under the MIT License.
The models are downloaded, not distributed here.
all-MiniLM-L6-v2 of sentence-transformers, Qwen3-Embedding-0.6B of Qwen and SPLADE of prithivida are under the Apache License 2.0.
The ONNX export of Qwen3-Embedding-0.6B by onnx-community declares no license of its own.
[THIRD-PARTY-NOTICES.md](https://github.com/ChromaDotNet/ChromaDB.Embeddings/blob/main/THIRD-PARTY-NOTICES.md) lists the models and the packages that NuGet installs, with their licenses.

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

The tests of the hosted models need their files, which they never download:

```
bash build/fetch-hosted-models.sh qwen-fp16 qwen-fp32 splade
dotnet test tests/ChromaDotNet.Embeddings.Tests
```

`build/fetch-hosted-models.sh` puts the files in `artifacts/hosted-models`, in the layout of the cache, checked against their
hashes; without arguments it takes `qwen-fp16` and `splade`.
A test checks that the files, revisions and hashes of the script are those of the package.
A model without its files is skipped.
`CHROMADOTNET_EMBEDDINGS_HOSTED=qwen-fp16,splade` makes those models required, and skips the others: without their files, the
tests fail.
`CHROMADOTNET_EMBEDDINGS_HOSTED_CACHE` takes the files from another cache, and `CHROMADOTNET_EMBEDDINGS_HOSTED_MODELS` from a
folder with a folder for each repository, named like `onnx-community__Qwen3-Embedding-0.6B-ONNX`.
`build/python-hosted-reference.py`, with onnxruntime 1.24.4, computes the vectors of the hosted models in Python again, and compares
them with the reference.
With `--output <file>` it writes those of this machine instead, and `CHROMADOTNET_EMBEDDINGS_HOSTED_PYTHON=<file>` gives them to the
tests, which require the vectors of .NET to be the same, bit for bit.
`dotnet run -c Release --project tests/ChromaDotNet.Embeddings.Bench -- qwen-fp16` measures the speed of a hosted model:
`qwen-fp16`, `qwen-fp32` or `splade`.
