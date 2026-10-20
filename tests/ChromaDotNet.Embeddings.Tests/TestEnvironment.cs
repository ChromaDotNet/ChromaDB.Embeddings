using System.Reflection;
using System.Text.Json;
using NUnit.Framework;

namespace ChromaDotNet.Embeddings.Tests;

// The folders of the tests, all under artifacts/test-tmp of the repository, and the archive of Python already downloaded.
internal static class TestEnvironment
{
	public static string Root { get; } = Metadata("EmbeddingsTestRoot");

	// CHROMADOTNET_EMBEDDINGS_ARCHIVE, or artifacts/model/onnx.tar.gz, where build/fetch-model.sh puts it: onnx.tar.gz as Python
	// downloads it.
	public static string? Archive
	{
		get
		{
			var path = Environment.GetEnvironmentVariable("CHROMADOTNET_EMBEDDINGS_ARCHIVE") is { Length: > 0 } fromEnvironment ? fromEnvironment : Metadata("EmbeddingsArchive");
			return File.Exists(path) ? path : null;
		}
	}

	// Without the archive the tests of the download are skipped; in the CI they fail, as the CI downloads it first.
	public static string RequireArchive()
	{
		if (Archive is { } archive)
			return archive;
		var message = "No onnx.tar.gz for the local mirror: run build/fetch-model.sh, or set CHROMADOTNET_EMBEDDINGS_ARCHIVE to the archive " +
			"Python downloads (~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz).";
		if (Environment.GetEnvironmentVariable("CI") == "true")
			throw new AssertionException(message);
		throw new IgnoreException(message);
	}

	// The model cache shared by the tests that only need a model.
	public static string SharedCache => Path.Combine(Root, "model-cache");

	public static string NewDirectory(string name)
	{
		var path = Path.Combine(Root, name + "-" + Guid.NewGuid().ToString("N").Substring(0, 8));
		Directory.CreateDirectory(path);
		return path;
	}

	// An address where nothing listens: a port that was free a moment ago.
	public static Uri Nowhere()
	{
		var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
		listener.Start();
		var port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
		listener.Stop();
		return new Uri($"http://127.0.0.1:{port}/all-MiniLM-L6-v2/onnx.tar.gz");
	}

	private static string Metadata(string key) => typeof(TestEnvironment).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
		.First(a => a.Key == key).Value!;
}

// The texts and the reference of Python, from TestData.
internal static class Reference
{
	private static readonly Lazy<(string[] Kinds, string[] Texts)> CorpusData = new(() =>
	{
		using var document = JsonDocument.Parse(File.ReadAllBytes(DataPath("corpus.json")));
		var items = document.RootElement.EnumerateArray().ToArray();
		return (items.Select(e => e.GetProperty("kind").GetString()!).ToArray(), items.Select(e => e.GetProperty("text").GetString()!).ToArray());
	});

	private static readonly Lazy<(long[][] Ids, float[][] Vectors)> PythonData = new(() => Read(DataPath("python.json"), "batch32"));

	private static readonly Lazy<(string[] Texts, long[][] Ids, float[][] Vectors)> SpecialData = new(() =>
	{
		var (ids, vectors) = Read(DataPath("python-special.json"), "vectors");
		using var document = JsonDocument.Parse(File.ReadAllBytes(DataPath("python-special.json")));
		var texts = document.RootElement.GetProperty("texts").EnumerateArray().Select(e => e.GetString()!).ToArray();
		return (texts, ids, vectors);
	});

	public static string[] Kinds => CorpusData.Value.Kinds;

	public static string[] Texts => CorpusData.Value.Texts;

	public static long[][] PythonIds => PythonData.Value.Ids;

	public static float[][] PythonVectors => PythonData.Value.Vectors;

	public static (string[] Texts, long[][] Ids, float[][] Vectors) Special => SpecialData.Value;

	private static (long[][] Ids, float[][] Vectors) Read(string path, string vectorsProperty)
	{
		using var document = JsonDocument.Parse(File.ReadAllBytes(path));
		var root = document.RootElement;
		var ids = root.GetProperty("ids").EnumerateArray().Select(row => row.EnumerateArray().Select(x => x.GetInt64()).ToArray()).ToArray();
		var vectors = root.GetProperty(vectorsProperty).EnumerateArray().Select(row => row.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray();
		return (ids, vectors);
	}

	private static string DataPath(string name) => Path.Combine(AppContext.BaseDirectory, "TestData", name);
}

// One generator for the tests that only compute vectors, on the shared cache, filled from the archive of Python when it is there.
[SetUpFixture]
public sealed class SharedGenerator
{
	private static readonly SemaphoreSlim Gate = new(1, 1);
	private static ChromaDefaultEmbeddingGenerator? _generator;

	internal static async Task<ChromaDefaultEmbeddingGenerator> GetAsync()
	{
		await Gate.WaitAsync();
		try
		{
			if (_generator is not null)
				return _generator;
			var cache = TestEnvironment.SharedCache;
			Directory.CreateDirectory(cache);
			var archive = Path.Combine(cache, "onnx.tar.gz");
			if (!ModelCache.HasAllFiles(Path.Combine(cache, "onnx")) && !File.Exists(archive) && TestEnvironment.Archive is { } source)
				File.Copy(source, archive);
			_generator = await ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { CacheDirectory = cache });
			return _generator;
		}
		finally
		{
			Gate.Release();
		}
	}

	[OneTimeTearDown]
	public void TearDown() => _generator?.Dispose();
}
