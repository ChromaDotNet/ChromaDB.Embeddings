namespace ChromaDotNet.Embeddings.Server;

/// <summary>
/// The configuration of the service, from the section <c>Embeddings</c> of appsettings.json, from the variables of the environment
/// (<c>Embeddings__Models=qwen</c>) or from the command line (<c>--Embeddings:Models=qwen</c>). Nothing of it comes from a collection.
/// </summary>
public sealed class EmbeddingServerOptions
{
	public const string Section = "Embeddings";

	/// <summary>
	/// The models the service computes, separated by commas: <c>qwen</c>, <c>splade</c>, or <c>qwen,splade</c>, the default. Each is
	/// loaded at its first request.
	/// </summary>
	public string Models { get; set; } = QwenModel + "," + SpladeModel;

	/// <summary>Qwen3-Embedding-0.6B in float16, the default, as Chroma Cloud computes it, or in float32.</summary>
	public ChromaQwenVariant QwenVariant { get; set; } = ChromaQwenVariant.Float16;

	/// <summary>
	/// The folder of the cache, with a folder for each model: <c>~/.cache/chroma/onnx_models</c> by default, as for
	/// ChromaDotNet.Embeddings.
	/// </summary>
	public string? CacheDirectory { get; set; }

	/// <summary>A folder with the files of the variant of Qwen, in the layout of the repository on Hugging Face; nothing is downloaded.</summary>
	public string? QwenModelDirectory { get; set; }

	/// <summary>A folder with the files of SPLADE, in the layout of the repository on Hugging Face; nothing is downloaded.</summary>
	public string? SpladeModelDirectory { get; set; }

	/// <summary>A mirror of Hugging Face, in place of <c>https://huggingface.co/</c>.</summary>
	public Uri? DownloadBaseUri { get; set; }

	/// <summary>Never downloads: a model whose files are not there gives an error at its first request.</summary>
	public bool Offline { get; set; }

	/// <summary>The threads of ONNX Runtime for each model: half the processors by default.</summary>
	public int? IntraOpNumThreads { get; set; }

	/// <summary>
	/// The requests that wait for a model while it computes another one; one more gets 503. A model computes one request at a time,
	/// and Qwen one text at a time: a text of 8192 tokens takes about 12 GB.
	/// </summary>
	public int QueueLimit { get; set; } = 16;

	/// <summary>
	/// The key the clients send in <c>x-chroma-token</c>, as they send the key of Chroma Cloud. It is required when the service listens
	/// on an address other than the loopback one, unless <see cref="AllowNoApiKey"/> is set.
	/// </summary>
	public string? ApiKey { get; set; }

	/// <summary>Listens on any address without a key: every client that reaches the port can use the service.</summary>
	public bool AllowNoApiKey { get; set; }

	public const string QwenModel = "qwen";
	public const string SpladeModel = "splade";
}
