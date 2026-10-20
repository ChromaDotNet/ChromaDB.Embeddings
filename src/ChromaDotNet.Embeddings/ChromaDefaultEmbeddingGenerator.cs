using System.Diagnostics;
#if NET
using ElBruno.LocalEmbeddings;
#endif
using Microsoft.Extensions.AI;

namespace ChromaDotNet.Embeddings;

/// <summary>
/// The default embedding function of Chroma, as the Python client computes it (<c>ONNXMiniLM_L6_V2</c> of chromadb): all-MiniLM-L6-v2
/// in ONNX, texts truncated to 256 tokens, mean pooling with the attention mask, L2 normalization, 384 dimensions.
/// </summary>
/// <remarks>
/// Create it with <see cref="CreateAsync"/>. The computation runs on the CPU in the calling thread, in batches of 32 texts; concurrent
/// calls run one after the other.
/// </remarks>
public sealed class ChromaDefaultEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
	/// <summary>
	/// The name Python stores in the configuration of a collection created without an embedding function, which then uses this one:
	/// <c>{"type": "known", "name": "default", "config": {}}</c>.
	/// </summary>
	public const string EmbeddingFunctionName = "default";

	/// <summary>The model.</summary>
	public const string ModelId = "all-MiniLM-L6-v2";

	/// <summary>The dimensions of each vector.</summary>
	public const int Dimensions = 384;

	/// <summary>The tokens of a text the model reads, <c>[CLS]</c> and <c>[SEP]</c> included: the rest of a longer text is ignored.</summary>
	public const int MaxTokens = 256;

	/// <summary>
	/// The distance function Python gives a collection created without an embedding function and without a space (chromadb 1.5.9:
	/// <c>DefaultEmbeddingFunction</c>, whose <c>default_space()</c> is the one of the base class). Python gives <c>cosine</c> only
	/// when <c>ONNXMiniLM_L6_V2</c> is passed explicitly, and then stores the name <c>onnx_mini_lm_l6_v2</c>.
	/// </summary>
	public const string DefaultSpace = "l2";

	private const int BatchSize = 32;

	private readonly ChromaTokenizer _tokenizer;
	private readonly OnnxEmbeddingModel _model;
	private readonly EmbeddingGeneratorMetadata _metadata = new("chroma", providerUri: null, defaultModelId: ModelId, defaultModelDimensions: Dimensions);
	private readonly object _gate = new();
	private bool _disposed;

	private ChromaDefaultEmbeddingGenerator(ChromaTokenizer tokenizer, OnnxEmbeddingModel model, string modelDirectory, int intraOpNumThreads)
	{
		_tokenizer = tokenizer;
		_model = model;
		ModelDirectory = modelDirectory;
		IntraOpNumThreads = intraOpNumThreads;
	}

	// The folder of the files loaded, and the threads of ONNX Runtime, for the tests.
	internal string ModelDirectory { get; }

	internal int IntraOpNumThreads { get; }

	internal ChromaTokenizer Tokenizer => _tokenizer;

	/// <summary>
	/// Downloads the model if it is not in the cache (about 83 MB, checked against its SHA-256 hash), then loads the tokenizer and the
	/// ONNX model.
	/// </summary>
	/// <param name="options">The cache folder, a folder already extracted, a mirror, the threads.</param>
	/// <param name="cancellationToken">Cancels the download.</param>
	/// <exception cref="InvalidOperationException">The model is not in the cache and cannot be downloaded.</exception>
	/// <exception cref="InvalidDataException">The downloaded file does not have the expected hash; it is not kept.</exception>
	public static async Task<ChromaDefaultEmbeddingGenerator> CreateAsync(ChromaDefaultEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
	{
		options ??= new ChromaDefaultEmbeddingOptions();
		var threads = options.IntraOpNumThreads ?? Math.Max(1, Environment.ProcessorCount / 2);
		if (threads < 1)
			throw new ArgumentOutOfRangeException(nameof(options), options.IntraOpNumThreads, "ChromaDefaultEmbeddingOptions.IntraOpNumThreads must be at least 1.");

		var onnx = await ModelCache.EnsureAsync(options, cancellationToken).ConfigureAwait(false);
		cancellationToken.ThrowIfCancellationRequested();

		var stopwatch = Stopwatch.StartNew();
		ChromaTokenizer? tokenizer = null;
		OnnxEmbeddingModel? model = null;
		try
		{
			tokenizer = ChromaTokenizer.Load(onnx, options.Logger);
			model = new OnnxEmbeddingModel();
			// One batch at a time, its operators on the threads given: faster than the parallel execution of ONNX Runtime here.
			model.Load(Path.Combine(onnx, "model.onnx"), normalizeEmbeddings: true, useParallelExecution: false, interOpNumThreads: 1, intraOpNumThreads: threads);
			if (model.EmbeddingDimension != Dimensions)
				throw new InvalidDataException($"The model of {onnx} gives {model.EmbeddingDimension} dimensions, not {Dimensions}.");
		}
		catch
		{
			tokenizer?.Dispose();
			model?.Dispose();
			throw;
		}
		if (options.Logger is { } logger)
			Log.Loaded(logger, onnx, threads, stopwatch.ElapsedMilliseconds);
		return new ChromaDefaultEmbeddingGenerator(tokenizer, model, onnx, threads);
	}

	/// <inheritdoc />
	/// <remarks>
	/// The vectors are those of Python for the same texts. <see cref="EmbeddingGenerationOptions.Dimensions"/>, when set, must be 384;
	/// <see cref="GeneratedEmbeddings{TEmbedding}.Usage"/> has the tokens read, <c>[CLS]</c> and <c>[SEP]</c> included.
	/// </remarks>
	public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
	{
		Compat.NotNull(values, nameof(values));
		if (options?.Dimensions is { } dimensions && dimensions != Dimensions)
			throw new ArgumentException($"{ModelId} has {Dimensions} dimensions, not {dimensions}.", nameof(options));
		var texts = values as IReadOnlyList<string> ?? values.ToList();
		var embeddings = new List<Embedding<float>>(texts.Count);
		long tokens = 0;
		lock (_gate)
		{
			Compat.ThrowIfDisposed(_disposed, this);
			for (var start = 0; start < texts.Count; start += BatchSize)
			{
				cancellationToken.ThrowIfCancellationRequested();
				var count = Math.Min(BatchSize, texts.Count - start);
				var ids = new long[count][];
				var masks = new long[count][];
				for (var i = 0; i < count; i++)
				{
					var text = texts[start + i] ?? throw new ArgumentException($"The text at {start + i} is null.", nameof(values));
					ids[i] = new long[MaxTokens];
					masks[i] = new long[MaxTokens];
					tokens += _tokenizer.Encode(text, ids[i], masks[i]);
				}
				var vectors = _model.GenerateEmbeddings(ids, masks, cancellationToken);
				var createdAt = DateTimeOffset.UtcNow;
				foreach (var vector in vectors)
					embeddings.Add(new Embedding<float>(vector) { ModelId = ModelId, CreatedAt = createdAt });
			}
		}
		return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(embeddings)
		{
			Usage = new UsageDetails { InputTokenCount = tokens, TotalTokenCount = tokens },
		});
	}

	/// <inheritdoc />
	/// <remarks>
	/// <see cref="EmbeddingGeneratorMetadata"/> has the provider <c>chroma</c>, the model <c>all-MiniLM-L6-v2</c> and 384 dimensions.
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

	/// <summary>Releases the ONNX session and the tokenizer.</summary>
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
}
