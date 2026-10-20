# Security

## Supported versions

Security fixes go into the latest release of the `ChromaDotNet.Embeddings` package on NuGet. Earlier versions do not get fixes:
update to the latest one.

## Reporting a vulnerability

Please do not report a vulnerability in a public issue, discussion or pull request.

Report it privately on GitHub instead: [Report a vulnerability](https://github.com/ChromaDotNet/ChromaDB.Embeddings/security/advisories/new), in the **Security** tab of this repository. Only the maintainers see the report.

Include the version of the package, the operating system, and the steps to reproduce the problem.

## What happens next

- We acknowledge the report within 7 days.
- We confirm or rule out the vulnerability within 14 days, and keep you informed of the progress.
- A confirmed vulnerability is fixed in a new release. A GitHub security advisory follows the release, with credit to you unless you prefer otherwise.
- Please keep the details private until the advisory is published, and for at most 90 days from the report.

## How the model is downloaded

- The package does not contain the model. The first start downloads `onnx.tar.gz` from the address of the Python client of Chroma,
  or from `DownloadUri`.
- The package contains the SHA-256 hash of the archive. It checks the archive before using it, and does not keep a file with another
  hash.
- It extracts only the six files of `onnx/`, by their exact names: no entry of the archive can write outside that folder.

## How the package is made

- Only the CI publishes the release and the package, from the version tags of this repository, with
  [NuGet trusted publishing](https://learn.microsoft.com/nuget/nuget-org/trusted-publishing): no NuGet API key is kept by a person or
  in the repository.
- The organization requires two-factor authentication.
- The workflows pin their actions to commit hashes, and the Python reference its packages to their hashes. Dependabot and CodeQL check the dependencies and the code.

## Verifying a release

Each GitHub release has the packages that the CI built, and their build provenance (`provenance-<tag>.intoto.jsonl`), attested by
the CI. nuget.org adds its repository signature (`.signature.p7s`) to the packages it serves.

With the GitHub CLI:

```
gh attestation verify ChromaDotNet.Embeddings.1.0.0-preview.1.nupkg --repo ChromaDotNet/ChromaDB.Embeddings --signer-workflow ChromaDotNet/ChromaDB.Embeddings/.github/workflows/ci.yml --source-ref refs/tags/v1.0.0-preview.1
```
