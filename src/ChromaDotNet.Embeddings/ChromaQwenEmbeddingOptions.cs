using Microsoft.Extensions.Logging;

namespace ChromaDotNet.Embeddings;

/// <summary>The ONNX export of Qwen3-Embedding-0.6B that <see cref="ChromaQwenEmbeddingGenerator"/> runs.</summary>
public enum ChromaQwenVariant
{
	/// <summary>The weights in float16: <c>onnx/model_fp16.onnx</c> and its data, about 1.2 GB.</summary>
	Float16,

	/// <summary>The weights in float32: <c>onnx/model.onnx</c> and its data, about 2.4 GB.</summary>
	[System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "The name of the precision, as Float16.")]
	Float32,
}

/// <summary>
/// The instructions of a task of Qwen3-Embedding, as the Python client of Chroma keeps them: one for the documents and one for the
/// queries. An empty instruction leaves the text as it is; another one makes it <c>Instruct: {instruction}\nQuery:{text}</c>.
/// </summary>
public sealed class ChromaQwenInstructions
{
	/// <summary>The instructions of a task.</summary>
	/// <param name="documents">The instruction of the documents, often empty.</param>
	/// <param name="query">The instruction of the queries.</param>
	public ChromaQwenInstructions(string documents, string query)
	{
		Documents = Compat.NotNull(documents, nameof(documents));
		Query = Compat.NotNull(query, nameof(query));
	}

	/// <summary>The instruction of the documents.</summary>
	public string Documents { get; }

	/// <summary>The instruction of the queries.</summary>
	public string Query { get; }
}

/// <summary>The options of <see cref="ChromaQwenEmbeddingGenerator.CreateAsync"/>.</summary>
public sealed class ChromaQwenEmbeddingOptions
{
	/// <summary>The export of the model: <see cref="ChromaQwenVariant.Float16"/> by default.</summary>
	public ChromaQwenVariant Variant { get; set; } = ChromaQwenVariant.Float16;

	/// <summary>
	/// The task, whose instructions apply to the documents and to the queries: <c>nl_to_code</c> by default, as for the collections of
	/// Chroma Cloud. A task that is null, empty, or not in <see cref="Instructions"/> has empty instructions, as in Python.
	/// </summary>
	public string? Task { get; set; } = ChromaQwenEmbeddingGenerator.DefaultTask;

	/// <summary>
	/// The instructions of each task, in place of those of the Python client, <see cref="ChromaQwenEmbeddingGenerator.DefaultInstructions"/>.
	/// </summary>
	public IReadOnlyDictionary<string, ChromaQwenInstructions>? Instructions { get; set; }

	/// <summary>
	/// The folder of the model in the cache, with a folder for each variant: <c>~/.cache/chroma/onnx_models/Qwen3-Embedding-0.6B</c> by
	/// default, next to the folder of the default model of Python.
	/// </summary>
	public string? CacheDirectory { get; set; }

	/// <summary>
	/// A folder with the files of the variant, in the layout of the repository on Hugging Face: <c>tokenizer.json</c> and the files of
	/// <c>onnx/</c>. With it, nothing is downloaded and <see cref="CacheDirectory"/> is not used.
	/// </summary>
	public string? ModelDirectory { get; set; }

	/// <summary>
	/// A mirror of Hugging Face, in place of <c>https://huggingface.co/</c>: the files are downloaded from
	/// <c>{DownloadBaseUri}{repository}/resolve/{revision}/{path}</c>, and must have the same SHA-256 hashes.
	/// </summary>
	public Uri? DownloadBaseUri { get; set; }

	/// <summary>The threads of ONNX Runtime for a text. The default is half of <see cref="Environment.ProcessorCount"/>, at least 1.</summary>
	public int? IntraOpNumThreads { get; set; }

	/// <summary>Receives the download, the load of the model, and the texts truncated.</summary>
	public ILogger? Logger { get; set; }
}
