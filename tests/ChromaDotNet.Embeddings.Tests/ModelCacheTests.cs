using System.Text;
using NUnit.Framework;

namespace ChromaDotNet.Embeddings.Tests;

// The download, the cache and the extraction, with a local mirror that serves the real onnx.tar.gz (83 MB, not in the repository).
[TestFixture]
public class ModelCacheTests
{
	private string _cache = null!;

	[SetUp]
	public void SetUp() => _cache = TestEnvironment.NewDirectory("cache");

	[TearDown]
	public void TearDown()
	{
		if (Directory.Exists(_cache))
			Directory.Delete(_cache, recursive: true);
	}

	// An empty cache: one download, the archive with its hash, the six files, no temporary file. Then a start from the cache, without
	// the server, gives the same vectors.
	[Test]
	public async Task DownloadsIntoAnEmptyCacheThenStartsFromItWithoutTheServer()
	{
		var archive = TestEnvironment.RequireArchive();
		float[] first;
		using (var mirror = new MirrorServer(file: archive))
		{
			using var generator = await ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { CacheDirectory = _cache, DownloadUri = mirror.Uri });
			Assert.That(mirror.Requests, Is.EqualTo(1));
			first = (await generator.GenerateAsync(["Hello, World!"]))[0].Vector.ToArray();
		}
		var hash = await ModelCache.HashFileAsync(Path.Combine(_cache, "onnx.tar.gz"), CancellationToken.None);
		Assert.Multiple(() =>
		{
			Assert.That(hash, Is.EqualTo(ModelCache.ArchiveSha256));
			Assert.That(ModelCache.ModelFiles.Where(name => !File.Exists(Path.Combine(_cache, "onnx", name))), Is.Empty);
			Assert.That(File.Exists(Path.Combine(_cache, "onnx.tar.gz.lock")), Is.True, "the lock file stays");
			Assert.That(File.Exists(Path.Combine(_cache, "onnx", "tokenizer.untruncated.json")), Is.True);
			Assert.That(TemporaryFiles(), Is.Empty);
		});

		using var again = await ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { CacheDirectory = _cache, DownloadUri = TestEnvironment.Nowhere() });
		Assert.That((await again.GenerateAsync(["Hello, World!"]))[0].Vector.ToArray(), Is.EqualTo(first));
	}

	// A file with another hash: InvalidDataException, and nothing kept.
	[Test]
	public void WrongHashIsNotKept()
	{
		var body = new byte[1 << 20];
		new Random(42).NextBytes(body);
		using var mirror = new MirrorServer(body: body);
		var e = Assert.ThrowsAsync<InvalidDataException>(() => ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { CacheDirectory = _cache, DownloadUri = mirror.Uri }));
		Assert.Multiple(() =>
		{
			Assert.That(e!.Message, Does.Contain(ModelCache.ArchiveSha256).And.Contain(mirror.Uri.ToString()).And.Contain("not kept"));
			Assert.That(File.Exists(Path.Combine(_cache, "onnx.tar.gz")), Is.False);
			Assert.That(Directory.Exists(Path.Combine(_cache, "onnx")), Is.False);
			Assert.That(TemporaryFiles(), Is.Empty);
		});
	}

	// Two starts at the same time on an empty cache: both work, and the archive is downloaded once.
	[Test]
	public async Task TwoConcurrentStartsDownloadOnce()
	{
		var archive = TestEnvironment.RequireArchive();
		using var mirror = new MirrorServer(file: archive);
		var options = new ChromaDefaultEmbeddingOptions { CacheDirectory = _cache, DownloadUri = mirror.Uri };
		var both = await Task.WhenAll(
			Task.Run(() => ChromaDefaultEmbeddingGenerator.CreateAsync(options)),
			Task.Run(() => ChromaDefaultEmbeddingGenerator.CreateAsync(options)));
		try
		{
			Assert.That(mirror.Requests, Is.EqualTo(1));
			var a = (await both[0].GenerateAsync(["Una mela rossa"]))[0].Vector.ToArray();
			var b = (await both[1].GenerateAsync(["Una mela rossa"]))[0].Vector.ToArray();
			Assert.That(a, Is.EqualTo(b));
			Assert.That(TemporaryFiles(), Is.Empty);
		}
		finally
		{
			foreach (var generator in both)
				generator.Dispose();
		}
	}

	// No server and an empty cache: an InvalidOperationException that names the address and the options to work without the network.
	[Test]
	public void NoServerAndEmptyCacheGiveAClearError()
	{
		var nowhere = TestEnvironment.Nowhere();
		var e = Assert.ThrowsAsync<InvalidOperationException>(() => ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { CacheDirectory = _cache, DownloadUri = nowhere }));
		TestContext.Out.WriteLine(e!.Message);
		Assert.Multiple(() =>
		{
			Assert.That(e.Message, Does.Contain(nowhere.ToString()));
			Assert.That(e.Message, Does.Contain("DownloadUri").And.Contain("ModelDirectory"));
			Assert.That(e.Message, Does.Contain(ModelCache.ArchiveSha256).And.Contain(_cache));
			Assert.That(TemporaryFiles(), Is.Empty);
		});
	}

	// The default address is the one of Python.
	[Test]
	public void DefaultDownloadUriIsThatOfPython() =>
		Assert.That(ModelCache.DefaultDownloadUri, Is.EqualTo("https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"));

	// The default cache is the folder of Python: ~/.cache/chroma/onnx_models/all-MiniLM-L6-v2.
	[Test]
	public void DefaultCacheDirectoryIsThatOfPython()
	{
		var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
		Assert.That(ModelCache.DefaultCacheDirectory(), Is.EqualTo(Path.Combine(home, ".cache", "chroma", "onnx_models", "all-MiniLM-L6-v2")));
	}

	// onnx.tar.gz already in the cache, as Python leaves it, without onnx/: extracted, nothing downloaded.
	[Test]
	public async Task ArchiveOfPythonIsExtractedWithoutTheNetwork()
	{
		File.Copy(TestEnvironment.RequireArchive(), Path.Combine(_cache, "onnx.tar.gz"));
		using var mirror = new MirrorServer(body: [1, 2, 3]);
		using var generator = await ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { CacheDirectory = _cache, DownloadUri = mirror.Uri });
		Assert.That(mirror.Requests, Is.Zero);
		Assert.That(ModelCache.HasAllFiles(Path.Combine(_cache, "onnx")), Is.True);
	}

	// An archive in the cache with another hash, as a download cut short: downloaded again.
	[Test]
	public async Task DamagedArchiveIsDownloadedAgain()
	{
		var archive = TestEnvironment.RequireArchive();
		File.WriteAllBytes(Path.Combine(_cache, "onnx.tar.gz"), Encoding.ASCII.GetBytes("not an archive"));
		using var mirror = new MirrorServer(file: archive);
		using var generator = await ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { CacheDirectory = _cache, DownloadUri = mirror.Uri });
		Assert.That(mirror.Requests, Is.EqualTo(1));
		Assert.That(new FileInfo(Path.Combine(_cache, "onnx.tar.gz")).Length, Is.EqualTo(83_178_821));
	}

	// ModelDirectory: the files there, no download, the cache folder not created.
	[Test]
	public async Task ModelDirectoryNeedsNoCacheAndNoNetwork()
	{
		var shared = await SharedGenerator.GetAsync();
		var cache = Path.Combine(_cache, "unused");
		using var generator = await ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions
		{
			ModelDirectory = shared.ModelDirectory,
			CacheDirectory = cache,
			DownloadUri = TestEnvironment.Nowhere(),
		});
		Assert.That(generator.ModelDirectory, Is.EqualTo(shared.ModelDirectory));
		Assert.That(Directory.Exists(cache), Is.False);
	}

	[Test]
	public void ModelDirectoryWithoutTheFilesGivesAClearError()
	{
		var e = Assert.ThrowsAsync<InvalidOperationException>(() => ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { ModelDirectory = _cache }));
		Assert.That(e!.Message, Does.Contain("ModelDirectory").And.Contain(_cache).And.Contain("model.onnx and tokenizer.json"));
	}

	// Only the six files of onnx/ are written: an entry that leaves the folder, or any other entry, is skipped.
	[Test]
	public async Task ExtractionWritesOnlyTheSixFilesOfOnnx()
	{
		var archive = Path.Combine(_cache, "crafted.tar.gz");
		static (string, byte[]) Entry(string name, string content) => (name, Encoding.UTF8.GetBytes(content));
		TarBuilder.Write(archive,
		[
			Entry("../evil.txt", "outside"),
			Entry("onnx/../../evil2.txt", "outside"),
			Entry("onnx/sub/model.onnx", "nested"),
			Entry("other/model.onnx", "other folder"),
			Entry("onnx/extra.txt", "not a model file"),
			.. ModelCache.ModelFiles.Select(name => Entry("onnx/" + name, "content of " + name)),
		]);
		var onnx = Path.Combine(_cache, "out", "onnx");
		await ModelCache.ExtractAsync(archive, onnx, logger: null, CancellationToken.None);
		var written = Directory.GetFiles(_cache, "*", SearchOption.AllDirectories).Select(p => p.Substring(_cache.Length + 1)).OrderBy(p => p, StringComparer.Ordinal).ToArray();
		Assert.That(written, Is.EqualTo(ModelCache.ModelFiles.Select(n => Path.Combine("out", "onnx", n)).Append("crafted.tar.gz").OrderBy(p => p, StringComparer.Ordinal).ToArray()));
		Assert.That(File.ReadAllText(Path.Combine(onnx, "model.onnx")), Is.EqualTo("content of model.onnx"));
		Assert.That(File.Exists(Path.Combine(TestEnvironment.Root, "evil.txt")), Is.False);
	}

	// An archive without all six files: InvalidDataException, and none of them kept.
	[Test]
	public void ExtractionNeedsAllSixFiles()
	{
		var archive = Path.Combine(_cache, "partial.tar.gz");
		TarBuilder.Write(archive, ("onnx/model.onnx", [1, 2, 3]));
		var onnx = Path.Combine(_cache, "onnx");
		var e = Assert.ThrowsAsync<InvalidDataException>(() => ModelCache.ExtractAsync(archive, onnx, logger: null, CancellationToken.None));
		Assert.That(e!.Message, Does.Contain("onnx/tokenizer.json"));
		Assert.That(Directory.GetFiles(onnx), Is.Empty);
	}

	// The copy of tokenizer.json without its truncation and padding of 128, the rest unchanged.
	[Test]
	public async Task TokenizerCopyHasNoTruncationAndNoPadding()
	{
		var shared = await SharedGenerator.GetAsync();
		var original = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllBytes(Path.Combine(shared.ModelDirectory, "tokenizer.json")))!.AsObject();
		var copy = System.Text.Json.Nodes.JsonNode.Parse(ChromaTokenizer.Untruncated(File.ReadAllBytes(Path.Combine(shared.ModelDirectory, "tokenizer.json"))))!.AsObject();
		Assert.Multiple(() =>
		{
			Assert.That(original["truncation"]?["max_length"]?.GetValue<int>(), Is.EqualTo(128));
			Assert.That(copy["truncation"], Is.Null);
			Assert.That(copy["padding"], Is.Null);
			Assert.That(copy.ContainsKey("truncation") && copy.ContainsKey("padding"), Is.True);
			foreach (var key in original.Select(p => p.Key).Where(k => k is not "truncation" and not "padding"))
				Assert.That(System.Text.Json.Nodes.JsonNode.DeepEquals(original[key], copy[key]), Is.True, key);
		});
	}

	private string[] TemporaryFiles() => Directory.GetFiles(_cache, "*.tmp", SearchOption.AllDirectories);
}
