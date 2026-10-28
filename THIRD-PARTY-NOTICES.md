# Third-party notices of ChromaDotNet.Embeddings

ChromaDotNet.Embeddings is under the MIT License, in [LICENSE](LICENSE).

The package contains its own assembly, the MSBuild targets for .NET Framework, the README, the icon and this file. It contains no
model and no native library. NuGet installs the packages below from nuget.org, each with its own license. The first start of a
generator downloads its model.

## The models

The models are not in the package or in this repository.

### The default model

The first start of `ChromaDefaultEmbeddingGenerator` downloads it from the address the Python client of Chroma uses:
https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz.

- all-MiniLM-L6-v2, of sentence-transformers: Apache License 2.0. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Its base model, nreimers/MiniLM-L6-H384-uncased: MIT License. https://huggingface.co/nreimers/MiniLM-L6-H384-uncased. It is a
  version of MiniLM of Microsoft: https://github.com/microsoft/unilm, MIT License.
- The ONNX export of Chroma, which makes the archive: Apache License 2.0. https://github.com/chroma-core/onnx-embedding

### The hosted models of Chroma Cloud

The first start of `ChromaQwenEmbeddingGenerator` and of `ChromaSpladeEmbeddingGenerator` downloads their files from Hugging Face,
at the revisions below. The licenses are those that the model cards declare.

- Qwen3-Embedding-0.6B, of Qwen: Apache License 2.0. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Its base model, Qwen/Qwen3-0.6B-Base: Apache License 2.0. https://huggingface.co/Qwen/Qwen3-0.6B-Base
- Its ONNX export, onnx-community/Qwen3-Embedding-0.6B-ONNX, at the revision c25a394dd583836952667c12f008335071b3f43d: the files
  `tokenizer.json`, `onnx/model_fp16.onnx` and `onnx/model_fp16.onnx_data`, or `onnx/model.onnx` and `onnx/model.onnx_data`. Its
  model card declares no license of its own: it is Qwen3-Embedding-0.6B with ONNX weights.
  https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX
- SPLADE, prithivida/Splade_PP_en_v1, at the revision 762be6a7206e2f299182705972a65e5c46e62be2: Apache License 2.0. The files
  `tokenizer.json` and `onnx/model.onnx`, the ONNX export of its repository. https://huggingface.co/prithivida/Splade_PP_en_v1

## The packages that NuGet installs

The versions are those of the package graph of this release. The licenses are those of the nuspec of each package.

### Both targets

| Package | Version | License | Project |
|---|---|---|---|
| Tokenizers.DotNet | 1.4.1 | MIT | https://github.com/sappho192/Tokenizers.DotNet |
| Tokenizers.DotNet.runtime.win-x64, win-arm64 | 1.4.1 | MIT | https://github.com/sappho192/Tokenizers.DotNet |
| Microsoft.ML.OnnxRuntime, Microsoft.ML.OnnxRuntime.Managed | 1.24.4 | MIT | https://github.com/microsoft/onnxruntime |
| Microsoft.Extensions.AI.Abstractions | 10.4.1 | MIT | https://github.com/dotnet/extensions |
| Microsoft.Extensions.Logging.Abstractions, Microsoft.Extensions.DependencyInjection.Abstractions | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| System.Diagnostics.DiagnosticSource | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| System.Text.Json, System.Text.Encodings.Web, System.IO.Pipelines | 10.0.4 | MIT | https://github.com/dotnet/runtime |
| System.Runtime.CompilerServices.Unsafe | 6.1.2 | MIT | https://github.com/dotnet/maintenance-packages |

The native libraries of Tokenizers.DotNet contain the tokenizers library of Hugging Face, version 0.22.2, under the Apache License
2.0: https://github.com/huggingface/tokenizers. ONNX Runtime lists the notices of its own dependencies in `ThirdPartyNotices.txt`,
in its packages.

### .NET 8 and later

| Package | Version | License | Project |
|---|---|---|---|
| ElBruno.LocalEmbeddings | 1.6.6 | MIT | https://github.com/elbruno/elbruno.localembeddings |
| ElBruno.HuggingFace.Downloader | 0.6.0 | MIT | https://github.com/elbruno/ElBruno.HuggingFace.Downloader |
| Tokenizers.DotNet.runtime.linux-x64, linux-arm64, osx-x64, osx-arm64 | 1.4.1 | MIT | https://github.com/sappho192/Tokenizers.DotNet |
| Microsoft.ML.Tokenizers | 2.0.0 | MIT | https://github.com/dotnet/machinelearning |
| Google.Protobuf | 3.30.2 | BSD-3-Clause | https://github.com/protocolbuffers/protobuf |
| Microsoft.Extensions.Configuration, .Abstractions, .Binder | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| Microsoft.Extensions.DependencyInjection | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| Microsoft.Extensions.Diagnostics, .Abstractions | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| Microsoft.Extensions.Http | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| Microsoft.Extensions.Logging | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| Microsoft.Extensions.Options, .ConfigurationExtensions | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| Microsoft.Extensions.Primitives | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| System.Numerics.Tensors | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| System.Memory | 4.5.5 | MIT | https://github.com/dotnet/corefx |

### .NET Framework 4.6.2 and later

| Package | Version | License | Project |
|---|---|---|---|
| SharpZipLib | 1.4.2 | MIT | https://github.com/icsharpcode/SharpZipLib |
| Microsoft.Bcl.AsyncInterfaces | 10.0.5 | MIT | https://github.com/dotnet/runtime |
| System.Buffers | 4.6.1 | MIT | https://github.com/dotnet/maintenance-packages |
| System.Memory | 4.6.3 | MIT | https://github.com/dotnet/maintenance-packages |
| System.Numerics.Vectors | 4.6.1 | MIT | https://github.com/dotnet/maintenance-packages |
| System.Threading.Tasks.Extensions | 4.6.3 | MIT | https://github.com/dotnet/maintenance-packages |
| System.ValueTuple | 4.6.1 | MIT | https://github.com/dotnet/maintenance-packages |

## The reference data of the tests

The vectors in `tests/ChromaDotNet.Embeddings.Tests/TestData` are outputs of the Python client of Chroma, chromadb 1.5.9 (Apache
License 2.0, https://github.com/chroma-core/chroma), with the model above, on texts of this repository. They are in the repository,
not in the package.

In `tests/ChromaDotNet.Embeddings.Tests/TestData/hosted`, `cloud-v1.json` and `cloud-v2.json` have the vectors that the embedding
service of Chroma Cloud, https://embed.trychroma.com/, returned on 25 October 2026 for texts of this repository. `python-onnx.json`
has the outputs of onnxruntime 1.24.4 in Python (MIT License, https://github.com/microsoft/onnxruntime), with the hosted models
above, on the same texts.

## No NOTICE file

Section 4(d) of the Apache License 2.0 asks a redistribution to carry the NOTICE file of the work, when the work has one. The
package and the repository redistribute no work under the Apache License 2.0: the models are downloaded, from Chroma and from Hugging
Face, and the tokenizers library comes in the packages of Tokenizers.DotNet. Also, sentence-transformers, the tokenizers library of
Hugging Face and the ONNX export of Chroma have no NOTICE file in their repositories, and neither do Qwen3-Embedding-0.6B, its base
model, its ONNX export and SPLADE. So this repository has no NOTICE file.
