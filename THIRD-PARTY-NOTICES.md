# Third-party notices of ChromaDotNet.Embeddings

ChromaDotNet.Embeddings is under the MIT License, in [LICENSE](LICENSE).

The package contains its own assembly, the MSBuild targets for .NET Framework, the README, the icon and this file. It contains no
model and no native library. NuGet installs the packages below from nuget.org, each with its own license. The first start downloads
the model.

## The model

The model is not in the package or in this repository. The first start downloads it from the address the Python client of Chroma
uses: https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz.

- all-MiniLM-L6-v2, of sentence-transformers: Apache License 2.0. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Its base model, nreimers/MiniLM-L6-H384-uncased: MIT License. https://huggingface.co/nreimers/MiniLM-L6-H384-uncased. It is a
  version of MiniLM of Microsoft: https://github.com/microsoft/unilm, MIT License.
- The ONNX export of Chroma, which makes the archive: Apache License 2.0. https://github.com/chroma-core/onnx-embedding

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
| ElBruno.LocalEmbeddings | 1.6.4 | MIT | https://github.com/elbruno/elbruno.localembeddings |
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

## No NOTICE file

Section 4(d) of the Apache License 2.0 asks a redistribution to carry the NOTICE file of the work, when the work has one. The
package and the repository redistribute no work under the Apache License 2.0: the model is downloaded from Chroma, and the
tokenizers library comes in the packages of Tokenizers.DotNet. Also, sentence-transformers, the tokenizers library of Hugging Face
and the ONNX export of Chroma have no NOTICE file in their repositories. So this repository has no NOTICE file.
