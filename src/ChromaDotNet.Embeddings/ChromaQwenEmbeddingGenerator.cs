using System.Collections.ObjectModel;
using System.Diagnostics;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;

namespace ChromaDotNet.Embeddings;

/// <summary>
/// The hosted embedding function <c>chroma-cloud-qwen</c> of Chroma Cloud, computed on this machine: Qwen3-Embedding-0.6B in ONNX, the
/// export of onnx-community, 1024 dimensions, the same vectors as the service. <see cref="GenerateAsync"/> embeds documents, with the
/// instruction of the documents of the task; <see cref="Queries"/> embeds queries, with the instruction of the queries.
/// </summary>
/// <remarks>
/// Create it with <see cref="CreateAsync"/>. Each text is computed alone, without truncation up to <see cref="MaxTokens"/> tokens, on
/// the CPU in the calling thread; concurrent calls run one after the other. The embedding is the last hidden state of the last token,
/// normalized to length 1.
/// </remarks>
public sealed class ChromaQwenEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
	/// <summary>The name of the embedding function of Python, <c>ChromaCloudQwenEmbeddingFunction</c>, in the configuration of a collection.</summary>
	public const string EmbeddingFunctionName = "chroma-cloud-qwen";

	/// <summary>The model, as Chroma Cloud names it.</summary>
	public const string ModelId = "Qwen/Qwen3-Embedding-0.6B";

	/// <summary>The dimensions of each vector.</summary>
	public const int Dimensions = 1024;

	/// <summary>
	/// The tokens of a text the model reads, <c>&lt;|endoftext|&gt;</c> included: its <c>max_position_embeddings</c>. A longer text is
	/// truncated, and the logger says so. The memory grows with the square of the tokens of a text.
	/// </summary>
	public const int MaxTokens = 32768;

	/// <summary>The task of the collections of Chroma Cloud, for natural language queries on code and documents.</summary>
	public const string DefaultTask = "nl_to_code";

	/// <summary>The distance function the embedding function of Python gives a new collection.</summary>
	public const string DefaultSpace = "cosine";

	private readonly TruncatingTokenizer _tokenizer;
	private readonly QwenOnnxModel _model;
	private readonly string _documentsInstruction;
	private readonly string _queryInstruction;
	private readonly ILogger? _logger;
	private readonly EmbeddingGeneratorMetadata _metadata = new("chroma", providerUri: null, defaultModelId: ModelId, defaultModelDimensions: Dimensions);
	private readonly object _gate = new();
	private bool _disposed;

	private ChromaQwenEmbeddingGenerator(TruncatingTokenizer tokenizer, QwenOnnxModel model, ChromaQwenVariant variant, string? task, IReadOnlyDictionary<string, ChromaQwenInstructions> instructions, ILogger? logger, string modelDirectory, int intraOpNumThreads)
	{
		_tokenizer = tokenizer;
		_model = model;
		_logger = logger;
		Variant = variant;
		Task = task;
		Instructions = instructions;
		var resolved = Resolve(task, instructions);
		_documentsInstruction = resolved.Documents;
		_queryInstruction = resolved.Query;
		ModelDirectory = modelDirectory;
		IntraOpNumThreads = intraOpNumThreads;
		Queries = new QueryGenerator(this);
	}

	/// <summary>
	/// The instructions of the Python client of Chroma: for <c>nl_to_code</c>, none for the documents, and "Given a question about coding,
	/// retrieval code or passage that can solve user's question" for the queries.
	/// </summary>
	public static IReadOnlyDictionary<string, ChromaQwenInstructions> DefaultInstructions { get; } = new ReadOnlyDictionary<string, ChromaQwenInstructions>(
		new Dictionary<string, ChromaQwenInstructions>
		{
			[DefaultTask] = new(string.Empty, "Given a question about coding, retrieval code or passage that can solve user's question"),
		});

	/// <summary>
	/// The generator of the queries: the same model, with the instruction of the queries of the task. It belongs to this generator: its
	/// <see cref="IDisposable.Dispose"/> does nothing, and it stops working when this generator is disposed.
	/// </summary>
	public IEmbeddingGenerator<string, Embedding<float>> Queries { get; }

	/// <summary>The export of the model loaded.</summary>
	public ChromaQwenVariant Variant { get; }

	/// <summary>The task, as in the options.</summary>
	public string? Task { get; }

	/// <summary>The instructions of each task, as in the options, or <see cref="DefaultInstructions"/>.</summary>
	public IReadOnlyDictionary<string, ChromaQwenInstructions> Instructions { get; }

	// The folder of the files loaded, the threads, and the instructions applied, for the tests.
	internal string ModelDirectory { get; }

	internal int IntraOpNumThreads { get; }

	internal string DocumentsInstruction => _documentsInstruction;

	internal string QueryInstruction => _queryInstruction;

	internal TruncatingTokenizer Tokenizer => _tokenizer;

	/// <summary>
	/// Downloads the files of the variant if they are not in the cache (about 1.2 GB in float16, 2.4 GB in float32, from Hugging Face,
	/// checked against their SHA-256 hashes), then loads the tokenizer and the ONNX model.
	/// </summary>
	/// <param name="options">The variant, the task, the cache folder, a folder with the files, a mirror, the threads.</param>
	/// <param name="cancellationToken">Cancels the download.</param>
	/// <exception cref="InvalidOperationException">The files are not in the cache and cannot be downloaded.</exception>
	/// <exception cref="InvalidDataException">A downloaded file does not have the expected hash; it is not kept.</exception>
	/// <exception cref="PlatformNotSupportedException">The process is a 32-bit process.</exception>
	public static async Task<ChromaQwenEmbeddingGenerator> CreateAsync(ChromaQwenEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
	{
		options ??= new ChromaQwenEmbeddingOptions();
		ChromaDefaultEmbeddingGenerator.Check64BitProcess(Environment.Is64BitProcess);
		// The options are read once, before the download: a change to them meanwhile does not reach the generator.
		var variant = options.Variant;
		var task = options.Task;
		var logger = options.Logger;
		var threads = options.IntraOpNumThreads ?? Math.Max(1, Environment.ProcessorCount / 2);
		if (threads < 1)
			throw new ArgumentOutOfRangeException(nameof(options), options.IntraOpNumThreads, "ChromaQwenEmbeddingOptions.IntraOpNumThreads must be at least 1.");
		var model = HostedModels.Qwen(variant);
		// A copy: the instructions of the generator do not change with the dictionary of the options.
		var instructions = options.Instructions is null ? DefaultInstructions
			: new ReadOnlyDictionary<string, ChromaQwenInstructions>(options.Instructions.ToDictionary(pair => pair.Key, pair => pair.Value ?? throw new ArgumentException($"The instructions of the task {pair.Key} are null.", nameof(options)), StringComparer.Ordinal));

		var directory = await HuggingFaceCache.EnsureAsync(model, options.CacheDirectory, options.ModelDirectory, options.DownloadBaseUri, logger, cancellationToken).ConfigureAwait(false);
		cancellationToken.ThrowIfCancellationRequested();

		var stopwatch = Stopwatch.StartNew();
		TruncatingTokenizer? tokenizer = null;
		QwenOnnxModel? onnx = null;
		try
		{
			tokenizer = TruncatingTokenizer.Load(directory, MaxTokens, logger);
			onnx = QwenOnnxModel.Load(HuggingFaceCache.LocalPath(directory, HostedModels.QwenModelFile(variant)), threads);
			if (onnx.Dimensions != Dimensions)
				throw new InvalidDataException($"The model of {directory} gives {onnx.Dimensions} dimensions, not {Dimensions}.");
		}
		catch
		{
			tokenizer?.Dispose();
			onnx?.Dispose();
			throw;
		}
		if (logger is not null)
			Log.Loaded(logger, directory, threads, stopwatch.ElapsedMilliseconds);
		return new ChromaQwenEmbeddingGenerator(tokenizer, onnx, variant, task, instructions, logger, directory, threads);
	}

	/// <summary>The embeddings of documents, with the instruction of the documents of the task.</summary>
	/// <param name="values">The documents.</param>
	/// <param name="options"><see cref="EmbeddingGenerationOptions.Dimensions"/>, when set, must be 1024.</param>
	/// <param name="cancellationToken">Cancels the computation, also within a text.</param>
	/// <returns>The embeddings, and in <see cref="GeneratedEmbeddings{TEmbedding}.Usage"/> the tokens read.</returns>
	public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default) =>
		Generate(values, _documentsInstruction, options, cancellationToken);

	/// <inheritdoc />
	/// <remarks>
	/// <see cref="EmbeddingGeneratorMetadata"/> has the provider <c>chroma</c>, the model <c>Qwen/Qwen3-Embedding-0.6B</c> and 1024
	/// dimensions.
	/// </remarks>
	public object? GetService(Type serviceType, object? serviceKey = null)
	{
		Compat.NotNull(serviceType, nameof(serviceType));
		if (serviceKey is not null)
			return null;
		if (serviceType == typeof(EmbeddingGeneratorMetadata))
			return _metadata;
		return serviceType.IsInstanceOfType(this) ? this : null;
	}

	/// <summary>Releases the ONNX session and the tokenizer; <see cref="Queries"/> stops working too.</summary>
	public void Dispose()
	{
		lock (_gate)
		{
			if (_disposed)
				return;
			_disposed = true;
			_model.Dispose();
			_tokenizer.Dispose();
		}
	}

	// The text the model reads: as it is with an empty instruction, as the service does for the documents of nl_to_code; otherwise the
	// format of Qwen3-Embedding, with no space after "Query:", as the service builds it.
	internal static string Format(string instruction, string text) => instruction.Length == 0 ? text : $"Instruct: {instruction}\nQuery:{text}";

	// The instructions of the documents and of the queries, as Python chooses them: those of the task, or none for a task that is null,
	// empty, or not in the dictionary.
	internal static ChromaQwenInstructions Resolve(string? task, IReadOnlyDictionary<string, ChromaQwenInstructions> instructions) =>
		!string.IsNullOrEmpty(task) && instructions.TryGetValue(task!, out var found) ? found : NoInstructions;

	private static readonly ChromaQwenInstructions NoInstructions = new(string.Empty, string.Empty);

	private Task<GeneratedEmbeddings<Embedding<float>>> Generate(IEnumerable<string> values, string instruction, EmbeddingGenerationOptions? options, CancellationToken cancellationToken)
	{
		Compat.NotNull(values, nameof(values));
		if (options?.Dimensions is { } dimensions && dimensions != Dimensions)
			throw new ArgumentException($"{ModelId} gives {Dimensions} dimensions, not {dimensions}.", nameof(options));
		var texts = values as IReadOnlyList<string> ?? values.ToList();
		var embeddings = new List<Embedding<float>>(texts.Count);
		long tokens = 0;
		lock (_gate)
		{
			Compat.ThrowIfDisposed(_disposed, this);
			for (var i = 0; i < texts.Count; i++)
			{
				cancellationToken.ThrowIfCancellationRequested();
				var text = texts[i] ?? throw new ArgumentException($"The text at {i} is null.", nameof(values));
				var ids = _tokenizer.Encode(Format(instruction, text), out var count);
				if (count > ids.Length && _logger is not null)
					Log.Truncated(_logger, i, count, ids.Length);
				tokens += ids.Length;
				var vector = _model.Embed(ids, cancellationToken);
				embeddings.Add(new Embedding<float>(vector) { ModelId = ModelId, CreatedAt = DateTimeOffset.UtcNow });
			}
		}
		return System.Threading.Tasks.Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(embeddings)
		{
			Usage = new UsageDetails { InputTokenCount = tokens, TotalTokenCount = tokens },
		});
	}

	// The queries: the model of the generator, with the instruction of the queries.
	private sealed class QueryGenerator : IEmbeddingGenerator<string, Embedding<float>>
	{
		private readonly ChromaQwenEmbeddingGenerator _owner;

		public QueryGenerator(ChromaQwenEmbeddingGenerator owner) => _owner = owner;

		public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default) =>
			_owner.Generate(values, _owner._queryInstruction, options, cancellationToken);

		public object? GetService(Type serviceType, object? serviceKey = null)
		{
			Compat.NotNull(serviceType, nameof(serviceType));
			if (serviceKey is not null)
				return null;
			if (serviceType == typeof(EmbeddingGeneratorMetadata))
				return _owner._metadata;
			if (serviceType.IsInstanceOfType(this))
				return this;
			return serviceType.IsInstanceOfType(_owner) ? _owner : null;
		}

		// The model belongs to the generator of the documents.
		public void Dispose()
		{
		}
	}
}
