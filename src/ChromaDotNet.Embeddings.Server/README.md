# ChromaDotNet.Embeddings.Server

A service that computes `chroma-cloud-qwen` and `chroma-cloud-splade` as the embedding service of Chroma Cloud does, with the same requests and responses. The clients of Chroma that read `CHROMA_EMBED_URL` send their texts to it in place of `https://embed.trychroma.com`, with no change to the code or to the collections.

It runs Qwen3-Embedding-0.6B and SPLADE (prithivida/Splade_PP_en_v1) on the CPU with [ChromaDotNet.Embeddings](https://www.nuget.org/packages/ChromaDotNet.Embeddings). The service is separate from any database: the clients compute the embeddings and send them to their server, Chroma, VecturaDB or the distributed Chroma.

## Start it

As a dotnet tool, on the loopback address:

```
dotnet tool install --global ChromaDotNet.Embeddings.Server --prerelease
chromadotnet-embed
```

It listens on `http://127.0.0.1:8080`. Each model is downloaded from Hugging Face at its first request, 1.2 GB for Qwen in float16 and 533 MB for SPLADE, into `~/.cache/chroma/onnx_models`.

With Docker, for Linux x64 and arm64:

```
docker run -p 127.0.0.1:8080:8080 -e Embeddings__ApiKey=<key> -v chroma-embeddings:/data ghcr.io/chromadotnet/embeddings-server
```

The models go into the volume at `/data`.

## Point the clients at it

```
export CHROMA_EMBED_URL=http://127.0.0.1:8080
export CHROMA_API_KEY=<key>
```

The clients of Python, JavaScript and Rust read `CHROMA_EMBED_URL`, and send `CHROMA_API_KEY` in `x-chroma-token`, as for Chroma Cloud.

## The key

- When the service listens only on the loopback address (`127.0.0.1`, `::1`, `localhost`), no key is needed.
- On any other address, as in the container, it does not start without `Embeddings:ApiKey`, the key the clients send in `x-chroma-token`. A request without it gets 401.
- `Embeddings:AllowNoApiKey=true` serves every client that reaches the port, without a key.

## Configuration

All of it is in the service, never in a collection. Set it in `appsettings.json`, in the environment (`Embeddings__Models=qwen`) or on the command line (`--Embeddings:Models=qwen`).

| Setting | Default | |
|---|---|---|
| `Embeddings:Models` | `qwen,splade` | The models it computes. |
| `Embeddings:QwenVariant` | `Float16` | Qwen in float16, as Chroma Cloud, or `Float32`. |
| `Embeddings:CacheDirectory` | `~/.cache/chroma/onnx_models` | The cache, with a folder for each model. |
| `Embeddings:QwenModelDirectory`, `Embeddings:SpladeModelDirectory` | | A folder with the files of the model, in the layout of the repository on Hugging Face; nothing is downloaded. |
| `Embeddings:DownloadBaseUri` | `https://huggingface.co/` | A mirror of Hugging Face. |
| `Embeddings:Offline` | `false` | Never downloads: a model whose files are missing gives an error. |
| `Embeddings:IntraOpNumThreads` | half the processors | The threads of ONNX Runtime for each model. |
| `Embeddings:QueueLimit` | `16` | The requests that wait for a model; one more gets 503. |
| `Embeddings:ApiKey` | | The key of `x-chroma-token`. |
| `Embeddings:AllowNoApiKey` | `false` | No key on an address other than the loopback one. |
| `urls` | `http://127.0.0.1:8080` | The addresses it listens on. |

A model computes one request at a time, and Qwen one text at a time. Qwen in float16 takes about 2.2 GB of memory for a text of 512 tokens, 5.3 GB for 4096 and 12.3 GB for 8192, in our tests.

## Differences from Chroma Cloud

- **Qwen refuses a text of more than 8192 tokens**, counting its instruction and `<|endoftext|>`, with 413 and a message that says which text. Chroma Cloud computes texts up to 32768 tokens. Beyond 8192 tokens the attention mask of the ONNX export needs more than 4 GiB.
- SPLADE truncates a text at 512 tokens, as Chroma Cloud does.
- The vectors are those of Chroma Cloud within the tolerances of ChromaDotNet.Embeddings: in our tests, against the vectors Chroma Cloud gave on 25 October 2026, 1 − cosine at most 3.82e-6 for Qwen in float16, and values up to 0.045 apart for SPLADE.

## Notes on the clients

- The JavaScript client `@chroma-core/chroma-cloud-splade` 0.1.12 does not send `fetch_tokens`, so its `includeTokens` gives no labels, also with Chroma Cloud.
- The JavaScript client `@chroma-core/chroma-cloud-qwen` 0.1.12 shows "Failed to generate embeddings" for any error, and leaves out the message of the service.

## The requests

| | Request | Response |
|---|---|---|
| Qwen | `POST /` with `{"instructions": "...", "texts": [...]}` | `{"embeddings": [[...]], "num_tokens": n}` |
| SPLADE | `POST /embed_sparse` with `{"texts": [...], "task": "", "target": "", "fetch_tokens": "false"}` | `{"embeddings": [{"indices": [...], "values": [...], "labels": null}]}` |
| Error | | `{"error": "..."}`, with 400, 401, 404, 413, 500 or 503 |

The header `x-chroma-embedding-model`, when present, must name the model of the endpoint. `GET /healthz` says which models are loaded.

The service is provided as is. Please open an issue for anything that does not work as described.
