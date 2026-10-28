using Microsoft.Extensions.Logging;

namespace ChromaDotNet.Embeddings;

/// <summary>The options of <see cref="ChromaSpladeEmbeddingGenerator.CreateAsync"/>.</summary>
public sealed class ChromaSpladeEmbeddingOptions
{
	/// <summary>
	/// The folder of the model in the cache: <c>~/.cache/chroma/onnx_models/Splade_PP_en_v1</c> by default, next to the folder of the
	/// default model of Python.
	/// </summary>
	public string? CacheDirectory { get; set; }

	/// <summary>
	/// A folder with the files of the model, in the layout of the repository on Hugging Face: <c>tokenizer.json</c> and
	/// <c>onnx/model.onnx</c>. With it, nothing is downloaded and <see cref="CacheDirectory"/> is not used.
	/// </summary>
	public string? ModelDirectory { get; set; }

	/// <summary>
	/// A mirror of Hugging Face, in place of <c>https://huggingface.co/</c>: the files are downloaded from
	/// <c>{DownloadBaseUri}{repository}/resolve/{revision}/{path}</c>, and must have the same SHA-256 hashes.
	/// </summary>
	public Uri? DownloadBaseUri { get; set; }

	/// <summary>The threads of ONNX Runtime for a text. The default is half of <see cref="Environment.ProcessorCount"/>, at least 1.</summary>
	public int? IntraOpNumThreads { get; set; }

	/// <summary>
	/// The tokens of the vocabulary in <see cref="ChromaSparseEmbedding.Labels"/>, as <c>include_tokens</c> of the Python client. False by
	/// default: the labels are null, as the service of Chroma Cloud returns them.
	/// </summary>
	public bool IncludeTokens { get; set; }

	/// <summary>Receives the download and the load of the model.</summary>
	public ILogger? Logger { get; set; }
}
