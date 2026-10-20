# Contributing

Issues and pull requests are welcome. To report a vulnerability, follow [SECURITY.md](SECURITY.md) instead.

## Build and test

The commands are in [Build and tests](README.md#build-and-tests), in the README. `build/fetch-model.sh` comes first: the tests of the
download serve the real model archive from a local mirror.

## Pull requests

- A fix comes with a test that fails without it.
- New functionality comes with tests in the automated test suite, which the CI runs on every pull request.
- The vectors stay those of Python: the parity tests compare them with `tests/ChromaDotNet.Embeddings.Tests/TestData`, and
  `build/python-reference.py` computes that reference again with chromadb.
- The public API is the same on .NET 8 and on .NET Framework. It is listed in `PublicAPI.Shipped.txt` and
  `PublicAPI.Unshipped.txt`: a new public member goes in `PublicAPI.Unshipped.txt`, otherwise the build fails with RS0016.
- Public types and members have XML documentation, which goes in the package.
- The build treats warnings as errors.

The CI, in [.github/workflows/ci.yml](.github/workflows/ci.yml), runs the tests on Windows, Linux x64 and arm64, and macOS arm64, and
on .NET Framework 4.8 on Windows. Only the version tags publish the release and the package.
