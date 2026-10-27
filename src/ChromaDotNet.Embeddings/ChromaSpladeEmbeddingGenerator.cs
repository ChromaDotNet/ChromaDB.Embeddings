using System.Diagnostics;
using Microsoft.Extensions.AI;

namespace ChromaDotNet.Embeddings;

/// <summary>
/// The hosted sparse embedding function <c>chroma-cloud-splade</c> of Chroma Cloud, computed on this machine: SPLADE
/// (prithivida/Splade_PP_en_v1) in ONNX, the export of its repository. The service gives its values in float16, this generator in
/// float32. On the texts of the tests, the terms of this generator are among those of the service, which has a few more below 0.02, and
/// the values differ by less than 0.05: the service itself gives values about 0.01 apart to a text alone and to the same text in a batch.
/// </summary>
/// <remarks>
/// Create it with <see cref="CreateAsync"/>. Each text is computed alone, truncated to <see cref="MaxTokens"/> tokens as the service
/// does, on the CPU in the calling thread; concurrent calls run one after the other. The value of each token of the vocabulary is the
/// maximum over the tokens of the text of log(1 + ReLU(logit)); the embedding keeps the values above zero.
/// </remarks>
public sealed class ChromaSpladeEmbeddingGenerator : IEmbeddingGenerator<string, ChromaSparseEmbedding>
{
	/// <summary>The name of the embedding function of Python, <c>ChromaCloudSpladeEmbeddingFunction</c>, in the configuration of a collection.</summary>
	public const string EmbeddingFunctionName = "chroma-cloud-splade";

	/// <summary>The model, as Chroma Cloud names it.</summary>
	public const string ModelId = "prithivida/Splade_PP_en_v1";

	/// <summary>The size of the vocabulary, the dimensions of the space of the embeddings.</summary>
	public const int Dimensions = 30522;

	/// <summary>The tokens of a text the model reads, <c>[CLS]</c> and <c>[SEP]</c> included: the rest of a longer text is ignored.</summary>
	public const int MaxTokens = 512;

	private readonly TruncatingTokenizer _tokenizer;
	private readonly SpladeOnnxModel _model;
	private readonly string[]? _vocabulary;
	private readonly EmbeddingGeneratorMetadata _metadata = new("chroma", providerUri: null, defaultModelId: ModelId, defaultModelDimensions: Dimensions);
	private readonly object _gate = new();
	private bool _disposed;

	private ChromaSpladeEmbeddingGenerator(TruncatingTokenizer tokenizer, SpladeOnnxModel model, string[]? vocabulary, string modelDirectory, int intraOpNumThreads)
	{
		_tokenizer = tokenizer;
		_model = model;
		_vocabulary = vocabulary;
		ModelDirectory = modelDirectory;
		IntraOpNumThreads = intraOpNumThreads;
	}

	/// <summary>Whether the embeddings have the tokens of their indices in <see cref="ChromaSparseEmbedding.Labels"/>.</summary>
	public bool IncludeTokens => _vocabulary is not null;

	// The folder of the files loaded, and the threads, for the tests.
	internal string ModelDirectory { get; }

	internal int IntraOpNumThreads { get; }

	internal TruncatingTokenizer Tokenizer => _tokenizer;

	/// <summary>
	/// Downloads the files of the model if they are not in the cache (about 530 MB, from Hugging Face, checked against their SHA-256
	/// hashes), then loads the tokenizer and the ONNX model.
	/// </summary>
	/// <param name="options">The cache folder, a folder with the files, a mirror, the threads, the tokens.</param>
	/// <param name="cancellationToken">Cancels the download.</param>
	/// <exception cref="InvalidOperationException">The files are not in the cache and cannot be downloaded.</exception>
	/// <exception cref="InvalidDataException">A downloaded file does not have the expected hash; it is not kept.</exception>
	/// <exception cref="PlatformNotSupportedException">The process is a 32-bit process.</exception>
	public static async Task<ChromaSpladeEmbeddingGenerator> CreateAsync(ChromaSpladeEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
	{
		options ??= new ChromaSpladeEmbeddingOptions();
		ChromaDefaultEmbeddingGenerator.Check64BitProcess(Environment.Is64BitProcess);
		// The options are read once, before the download: a change to them meanwhile does not reach the generator.
		var includeTokens = options.IncludeTokens;
		var logger = options.Logger;
		var threads = options.IntraOpNumThreads ?? Math.Max(1, Environment.ProcessorCount / 2);
		if (threads < 1)
			throw new ArgumentOutOfRangeException(nameof(options), options.IntraOpNumThreads, "ChromaSpladeEmbeddingOptions.IntraOpNumThreads must be at least 1.");

		var directory = await HuggingFaceCache.EnsureAsync(HostedModels.Splade, options.CacheDirectory, options.ModelDirectory, options.DownloadBaseUri, logger, cancellationToken).ConfigureAwait(false);
		cancellationToken.ThrowIfCancellationRequested();

		var stopwatch = Stopwatch.StartNew();
		TruncatingTokenizer? tokenizer = null;
		SpladeOnnxModel? model = null;
		try
		{
			tokenizer = TruncatingTokenizer.Load(directory, MaxTokens, logger);
			model = SpladeOnnxModel.Load(HuggingFaceCache.LocalPath(directory, "onnx/model.onnx"), threads);
			if (model.VocabularySize != Dimensions)
				throw new InvalidDataException($"The model of {directory} gives {model.VocabularySize} dimensions, not {Dimensions}.");
			var vocabulary = includeTokens ? TokenizerFile.Vocabulary(directory, Dimensions) : null;
			if (logger is not null)
				Log.Loaded(logger, directory, threads, stopwatch.ElapsedMilliseconds);
			return new ChromaSpladeEmbeddingGenerator(tokenizer, model, vocabulary, directory, threads);
		}
		catch
		{
			tokenizer?.Dispose();
			model?.Dispose();
			throw;
		}
	}

	/// <summary>The sparse embeddings of texts, the same for documents and queries.</summary>
	/// <param name="values">The texts.</param>
	/// <param name="options"><see cref="EmbeddingGenerationOptions.Dimensions"/>, when set, must be 30522.</param>
	/// <param name="cancellationToken">Cancels the computation, also within a text.</param>
	/// <returns>The embeddings, and in <see cref="GeneratedEmbeddings{TEmbedding}.Usage"/> the tokens read.</returns>
	public Task<GeneratedEmbeddings<ChromaSparseEmbedding>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
	{
		Compat.NotNull(values, nameof(values));
		if (options?.Dimensions is { } dimensions && dimensions != Dimensions)
			throw new ArgumentException($"{ModelId} gives {Dimensions} dimensions, not {dimensions}.", nameof(options));
		var texts = values as IReadOnlyList<string> ?? values.ToList();
		var embeddings = new List<ChromaSparseEmbedding>(texts.Count);
		long tokens = 0;
		lock (_gate)
		{
			Compat.ThrowIfDisposed(_disposed, this);
			for (var i = 0; i < texts.Count; i++)
			{
				cancellationToken.ThrowIfCancellationRequested();
				var text = texts[i] ?? throw new ArgumentException($"The text at {i} is null.", nameof(values));
				var ids = _tokenizer.Encode(text, out _);
				tokens += ids.Length;
				var (indices, vector) = _model.Embed(ids, cancellationToken);
				var labels = _vocabulary is null ? null : Array.ConvertAll(indices, index => _vocabulary[index]);
				embeddings.Add(new ChromaSparseEmbedding(indices, vector, labels, Dimensions) { ModelId = ModelId, CreatedAt = DateTimeOffset.UtcNow });
			}
		}
		return Task.FromResult(new GeneratedEmbeddings<ChromaSparseEmbedding>(embeddings)
		{
			Usage = new UsageDetails { InputTokenCount = tokens, TotalTokenCount = tokens },
		});
	}

	/// <inheritdoc />
	/// <remarks>
	/// <see cref="EmbeddingGeneratorMetadata"/> has the provider <c>chroma</c>, the model <c>prithivida/Splade_PP_en_v1</c> and the 30522
	/// dimensions of its vocabulary.
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
