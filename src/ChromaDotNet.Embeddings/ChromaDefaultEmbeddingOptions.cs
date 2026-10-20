using Microsoft.Extensions.Logging;

namespace ChromaDotNet.Embeddings;

/// <summary>The options of <see cref="ChromaDefaultEmbeddingGenerator.CreateAsync"/>.</summary>
public sealed class ChromaDefaultEmbeddingOptions
{
	/// <summary>
	/// The folder of the model archive and of its extracted files. The default is the folder of the Python client,
	/// <c>~/.cache/chroma/onnx_models/all-MiniLM-L6-v2</c> (<c>%USERPROFILE%\.cache\chroma\onnx_models\all-MiniLM-L6-v2</c> on Windows),
	/// so that a model downloaded by Python is used here, and the other way round.
	/// </summary>
	public string? CacheDirectory { get; set; }

	/// <summary>
	/// A folder with the files already extracted from the archive (the <c>onnx/</c> folder: <c>model.onnx</c> and <c>tokenizer.json</c>
	/// at least). With it, nothing is downloaded and <see cref="CacheDirectory"/> is not used.
	/// </summary>
	public string? ModelDirectory { get; set; }

	/// <summary>
	/// The address of <c>onnx.tar.gz</c>, for a mirror. The default is the address the Python client uses,
	/// <c>https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz</c>. The file must have the same SHA-256 hash.
	/// </summary>
	public Uri? DownloadUri { get; set; }

	/// <summary>
	/// The threads of ONNX Runtime within one batch. The default is half of <see cref="Environment.ProcessorCount"/>, at least 1: on
	/// CPUs with simultaneous multithreading or with efficiency cores, more threads are slower.
	/// </summary>
	public int? IntraOpNumThreads { get; set; }

	/// <summary>Receives the download, the extraction and the load of the model.</summary>
	public ILogger? Logger { get; set; }
}
