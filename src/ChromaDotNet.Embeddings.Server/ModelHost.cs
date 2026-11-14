using Microsoft.Extensions.Options;

namespace ChromaDotNet.Embeddings.Server;

// A model loaded at its first request, which computes one request at a time; the others wait, up to the limit of the queue.
internal sealed class ModelHost<TGenerator>(Func<CancellationToken, Task<TGenerator>> create, int queueLimit) : IDisposable
	where TGenerator : class, IDisposable
{
	private readonly SemaphoreSlim _load = new(1, 1);
	private readonly SemaphoreSlim _compute = new(1, 1);
	private TGenerator? _generator;
	private int _waiting;

	public bool IsLoaded => Volatile.Read(ref _generator) is not null;

	// Runs the work with the model, or returns false when the queue is full.
	public async Task<(bool Accepted, TResult? Result)> RunAsync<TResult>(Func<TGenerator, CancellationToken, Task<TResult>> work, CancellationToken cancellationToken)
	{
		if (Interlocked.Increment(ref _waiting) > queueLimit + 1)
		{
			Interlocked.Decrement(ref _waiting);
			return (false, default);
		}
		try
		{
			await _compute.WaitAsync(cancellationToken).ConfigureAwait(false);
			try
			{
				var generator = await GetAsync(cancellationToken).ConfigureAwait(false);
				return (true, await work(generator, cancellationToken).ConfigureAwait(false));
			}
			finally
			{
				_compute.Release();
			}
		}
		finally
		{
			Interlocked.Decrement(ref _waiting);
		}
	}

	private async Task<TGenerator> GetAsync(CancellationToken cancellationToken)
	{
		if (Volatile.Read(ref _generator) is { } loaded)
			return loaded;
		await _load.WaitAsync(cancellationToken).ConfigureAwait(false);
		try
		{
			// A load that failed is tried again at the next request: the files may be there by then.
			return _generator ??= await create(cancellationToken).ConfigureAwait(false);
		}
		finally
		{
			_load.Release();
		}
	}

	public void Dispose()
	{
		_generator?.Dispose();
		_load.Dispose();
		_compute.Dispose();
	}
}

// The two models of the service, each created from the options when it is first used.
internal sealed class Models : IDisposable
{
	public Models(IOptions<EmbeddingServerOptions> options, ILoggerFactory loggerFactory)
	{
		var o = options.Value;
		Enabled = new HashSet<string>(o.Models.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(m => m.ToLowerInvariant()), StringComparer.Ordinal);
		if (Enabled.FirstOrDefault(m => m is not EmbeddingServerOptions.QwenModel and not EmbeddingServerOptions.SpladeModel) is { } unknown)
			throw new InvalidOperationException($"Embeddings:Models has {unknown}: the models are {EmbeddingServerOptions.QwenModel} and {EmbeddingServerOptions.SpladeModel}.");
		Qwen = new ModelHost<ChromaQwenEmbeddingGenerator>(cancellationToken =>
		{
			var model = HostedModels.Qwen(o.QwenVariant);
			if (o.Offline && o.QwenModelDirectory is null)
				CheckOffline(model, o.CacheDirectory, "QwenModelDirectory");
			return ChromaQwenEmbeddingGenerator.CreateAsync(new ChromaQwenEmbeddingOptions
			{
				Variant = o.QwenVariant,
				CacheDirectory = o.CacheDirectory is null ? null : Path.Combine(o.CacheDirectory, HostedModels.QwenFolder),
				ModelDirectory = o.QwenModelDirectory,
				DownloadBaseUri = o.DownloadBaseUri,
				IntraOpNumThreads = o.IntraOpNumThreads,
				Logger = loggerFactory.CreateLogger("ChromaDotNet.Embeddings.Qwen"),
			}, cancellationToken);
		}, o.QueueLimit);
		Splade = new ModelHost<ChromaSpladeEmbeddingGenerator>(cancellationToken =>
		{
			if (o.Offline && o.SpladeModelDirectory is null)
				CheckOffline(HostedModels.Splade, o.CacheDirectory, "SpladeModelDirectory");
			return ChromaSpladeEmbeddingGenerator.CreateAsync(new ChromaSpladeEmbeddingOptions
			{
				CacheDirectory = o.CacheDirectory is null ? null : Path.Combine(o.CacheDirectory, HostedModels.SpladeFolder),
				ModelDirectory = o.SpladeModelDirectory,
				DownloadBaseUri = o.DownloadBaseUri,
				IntraOpNumThreads = o.IntraOpNumThreads,
				// The labels are computed always, and left out of a response that does not ask for them.
				IncludeTokens = true,
				Logger = loggerFactory.CreateLogger("ChromaDotNet.Embeddings.Splade"),
			}, cancellationToken);
		}, o.QueueLimit);
	}

	public IReadOnlySet<string> Enabled { get; }

	public ModelHost<ChromaQwenEmbeddingGenerator> Qwen { get; }

	public ModelHost<ChromaSpladeEmbeddingGenerator> Splade { get; }

	private static void CheckOffline(HuggingFaceModel model, string? cacheDirectory, string option)
	{
		var root = Path.GetFullPath(cacheDirectory is null ? Path.Combine(ModelCache.ModelsRoot(model.OptionsName), model.Folder) : Path.Combine(cacheDirectory, model.Folder));
		var directory = model.Subfolder is null ? root : Path.Combine(root, model.Subfolder);
		if (!HuggingFaceCache.HasAllFiles(model, directory))
			throw new InvalidOperationException(
				$"Embeddings:Offline is set and {directory} does not have all the files of {model.Repository} at the revision {model.Revision}: " +
				$"put them there, set Embeddings:{option} to a folder with them, or turn Offline off to download them.");
	}

	public void Dispose()
	{
		Qwen.Dispose();
		Splade.Dispose();
	}
}
