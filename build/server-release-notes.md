Preview of ChromaDotNet.Embeddings.Server VERSION: a service that computes `chroma-cloud-qwen` and `chroma-cloud-splade` as the embedding service of Chroma Cloud does, with the same requests and responses, for the clients of Chroma that read `CHROMA_EMBED_URL`. It runs Qwen3-Embedding-0.6B and SPLADE on the CPU. Independent project. Not affiliated with, endorsed by or sponsored by Chroma Inc. CHROMA is a registered trademark of Chroma Inc., used here only to describe compatibility.

As a dotnet tool, on .NET 10, [ChromaDotNet.Embeddings.Server](https://www.nuget.org/packages/ChromaDotNet.Embeddings.Server/VERSION):

```
dotnet tool install --global ChromaDotNet.Embeddings.Server --version VERSION
chromadotnet-embed
```

As an image for Linux x64 and arm64:

```
docker run -p 127.0.0.1:8080:8080 -e Embeddings__ApiKey=<key> -v chroma-embeddings:/data ghcr.io/chromadotnet/embeddings-server:VERSION
```

Qwen refuses a text of more than 8192 tokens, where Chroma Cloud computes up to 32768. The configuration and the other differences are in the [README](https://github.com/ChromaDotNet/ChromaDB.Embeddings/blob/TAG/src/ChromaDotNet.Embeddings.Server/README.md). The service is provided as is.

The files of this release are the package of the tool that the CI built, and its build provenance. To check the package and the image:

```
gh attestation verify ChromaDotNet.Embeddings.Server.VERSION.nupkg --repo ChromaDotNet/ChromaDB.Embeddings
gh attestation verify oci://ghcr.io/chromadotnet/embeddings-server:VERSION --repo ChromaDotNet/ChromaDB.Embeddings
```
