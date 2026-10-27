using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using NUnit.Framework;

namespace ChromaDotNet.Embeddings.Tests;

// The download of the files of a hosted model from a mirror of Hugging Face on 127.0.0.1: a model of two small files with their hashes,
// then SPLADE with its real files.
[TestFixture]
public class HostedCacheTests
{
	private const string Repository = "owner/model";
	private const string Revision = "0123456789abcdef0123456789abcdef01234567";
	private static readonly byte[] Tokenizer = Encoding.UTF8.GetBytes("{\"model\": \"a tokenizer\"}");
	private static readonly byte[] Weights = Enumerable.Range(0, 1 << 16).Select(i => (byte)(i * 7)).ToArray();

	private string _cache = null!;

	[SetUp]
	public void SetUp() => _cache = TestEnvironment.NewDirectory("hosted-cache");

	[TearDown]
	public void TearDown()
	{
		if (Directory.Exists(_cache))
			Directory.Delete(_cache, recursive: true);
	}

	private static HuggingFaceModel Model(string? subfolder = "fp16") => new(Repository, Revision, "Model", subfolder, "TestOptions",
		new HuggingFaceFile("tokenizer.json", Sha256(Tokenizer), Tokenizer.Length),
		new HuggingFaceFile("onnx/model.onnx", Sha256(Weights), Weights.Length));

	private static string Route(string path) => $"/{Repository}/resolve/{Revision}/{path}";

	private static Dictionary<string, object> Routes(byte[]? weights = null) => new()
	{
		[Route("tokenizer.json")] = Tokenizer,
		[Route("onnx/model.onnx")] = weights ?? Weights,
	};

	// An empty cache: each file once, in the layout of the repository, in the folder of the variant, with the lock file and no
	// temporary file. Then the files of the cache, without the server.
	[Test]
	public async Task DownloadsTheFilesThenUsesTheCache()
	{
		var model = Model();
		string directory;
		using (var mirror = new MirrorServer(Routes()))
		{
			directory = await HuggingFaceCache.EnsureAsync(model, _cache, null, mirror.BaseUri, null, CancellationToken.None);
			Assert.Multiple(() =>
			{
				Assert.That(mirror.RequestsOf(Route("tokenizer.json")), Is.EqualTo(1));
				Assert.That(mirror.RequestsOf(Route("onnx/model.onnx")), Is.EqualTo(1));
			});
		}
		Assert.Multiple(() =>
		{
			Assert.That(directory, Is.EqualTo(Path.Combine(Path.GetFullPath(_cache), "fp16")));
			Assert.That(File.ReadAllBytes(Path.Combine(directory, "tokenizer.json")), Is.EqualTo(Tokenizer));
			Assert.That(File.ReadAllBytes(Path.Combine(directory, "onnx", "model.onnx")), Is.EqualTo(Weights));
			Assert.That(File.Exists(Path.Combine(directory, "download.lock")), Is.True, "the lock file stays");
			Assert.That(Temporary(), Is.Empty);
		});
		var again = await HuggingFaceCache.EnsureAsync(model, _cache, null, TestEnvironment.Nowhere(), null, CancellationToken.None);
		Assert.That(again, Is.EqualTo(directory));
	}

	// A model without a variant has its files in the folder of the model; only the file that is missing is downloaded.
	[Test]
	public async Task DownloadsOnlyTheMissingFile()
	{
		var model = Model(subfolder: null);
		File.WriteAllBytes(Path.Combine(_cache, "tokenizer.json"), Tokenizer);
		using var mirror = new MirrorServer(Routes());
		var directory = await HuggingFaceCache.EnsureAsync(model, _cache, null, mirror.BaseUri, null, CancellationToken.None);
		Assert.Multiple(() =>
		{
			Assert.That(directory, Is.EqualTo(Path.GetFullPath(_cache)));
			Assert.That(mirror.RequestsOf(Route("tokenizer.json")), Is.EqualTo(0));
			Assert.That(mirror.RequestsOf(Route("onnx/model.onnx")), Is.EqualTo(1));
		});
	}

	// A file of another size, like a copy cut short, counts as missing: it is downloaded again, and the other file is not.
	[Test]
	public async Task FileOfAnotherSizeIsDownloadedAgain()
	{
		var directory = Path.Combine(_cache, "fp16");
		Directory.CreateDirectory(Path.Combine(directory, "onnx"));
		File.WriteAllBytes(Path.Combine(directory, "tokenizer.json"), Tokenizer);
		File.WriteAllBytes(Path.Combine(directory, "onnx", "model.onnx"), Weights.Take(1000).ToArray());
		Assert.That(HuggingFaceCache.HasAllFiles(Model(), directory), Is.False);
		using var mirror = new MirrorServer(Routes());
		await HuggingFaceCache.EnsureAsync(Model(), _cache, null, mirror.BaseUri, null, CancellationToken.None);
		Assert.Multiple(() =>
		{
			Assert.That(mirror.RequestsOf(Route("tokenizer.json")), Is.EqualTo(0));
			Assert.That(mirror.RequestsOf(Route("onnx/model.onnx")), Is.EqualTo(1));
			Assert.That(File.ReadAllBytes(Path.Combine(directory, "onnx", "model.onnx")), Is.EqualTo(Weights));
			Assert.That(Temporary(), Is.Empty);
		});
	}

	// A file with another hash: InvalidDataException, and nothing kept of it.
	[Test]
	public void WrongHashIsNotKept()
	{
		var other = (byte[])Weights.Clone();
		other[100] ^= 1;
		using var mirror = new MirrorServer(Routes(other));
		var e = Assert.ThrowsAsync<InvalidDataException>(() => HuggingFaceCache.EnsureAsync(Model(), _cache, null, mirror.BaseUri, null, CancellationToken.None));
		Assert.Multiple(() =>
		{
			Assert.That(e!.Message, Does.Contain(Sha256(Weights)).And.Contain("not kept"));
			Assert.That(File.Exists(Path.Combine(_cache, "fp16", "onnx", "model.onnx")), Is.False);
			Assert.That(Temporary(), Is.Empty);
		});
	}

	// A file the mirror does not have: the message says what to do without the network.
	[Test]
	public void MissingFileSaysWhatToDo()
	{
		var routes = Routes();
		routes.Remove(Route("onnx/model.onnx"));
		using var mirror = new MirrorServer(routes);
		var e = Assert.ThrowsAsync<InvalidOperationException>(() => HuggingFaceCache.EnsureAsync(Model(), _cache, null, mirror.BaseUri, null, CancellationToken.None));
		Assert.That(e!.Message, Does.Contain("onnx/model.onnx").And.Contain(Repository).And.Contain(Revision).And.Contain("TestOptions.DownloadBaseUri")
			.And.Contain("ModelDirectory").And.Contain(Sha256(Weights)));
	}

	// A mirror under a path, written with or without the slash at the end.
	[Test]
	public async Task MirrorUnderAPath()
	{
		var routes = Routes().ToDictionary(pair => "/hf" + pair.Key, pair => pair.Value);
		using var mirror = new MirrorServer(routes);
		var directory = await HuggingFaceCache.EnsureAsync(Model(), _cache, null, new Uri(mirror.BaseUri, "hf"), null, CancellationToken.None);
		Assert.That(File.ReadAllBytes(Path.Combine(directory, "onnx", "model.onnx")), Is.EqualTo(Weights));
		Assert.That(Model().FileUri(new Uri(mirror.BaseUri, "hf/"), Model().Files[0]), Is.EqualTo(Model().FileUri(new Uri(mirror.BaseUri, "hf"), Model().Files[0])));
		Assert.That(Model().FileUri(null, Model().Files[1]).ToString(), Is.EqualTo($"https://huggingface.co/{Repository}/resolve/{Revision}/onnx/model.onnx"));
	}

	// Two calls at the same time download each file once: the second one waits for the lock, then finds the files.
	[Test]
	public async Task ConcurrentCallsDownloadOnce()
	{
		using var mirror = new MirrorServer(Routes());
		var calls = Enumerable.Range(0, 4).Select(_ => Task.Run(() => HuggingFaceCache.EnsureAsync(Model(), _cache, null, mirror.BaseUri, null, CancellationToken.None))).ToArray();
		var directories = await Task.WhenAll(calls);
		Assert.Multiple(() =>
		{
			Assert.That(directories.Distinct(), Has.Exactly(1).Items);
			Assert.That(mirror.RequestsOf(Route("tokenizer.json")), Is.EqualTo(1));
			Assert.That(mirror.RequestsOf(Route("onnx/model.onnx")), Is.EqualTo(1));
		});
	}

	// ModelDirectory is used as it is: nothing downloaded, and the files it lacks named with their sizes, also a file of another size.
	[Test]
	public async Task ModelDirectoryIsUsedAsItIs()
	{
		var folder = TestEnvironment.NewDirectory("hosted-model");
		try
		{
			File.WriteAllBytes(Path.Combine(folder, "tokenizer.json"), Tokenizer);
			var e = Assert.ThrowsAsync<InvalidOperationException>(() => HuggingFaceCache.EnsureAsync(Model(), null, folder, TestEnvironment.Nowhere(), null, CancellationToken.None));
			Assert.That(e!.Message, Does.Contain("TestOptions.ModelDirectory").And.Contain($"has no onnx/model.onnx of {Weights.Length} bytes:").And.Not.Contain("tokenizer.json"));
			Directory.CreateDirectory(Path.Combine(folder, "onnx"));
			File.WriteAllBytes(Path.Combine(folder, "onnx", "model.onnx"), Weights.Take(1000).ToArray());
			e = Assert.ThrowsAsync<InvalidOperationException>(() => HuggingFaceCache.EnsureAsync(Model(), null, folder, TestEnvironment.Nowhere(), null, CancellationToken.None));
			Assert.That(e!.Message, Does.Contain($"has no onnx/model.onnx of {Weights.Length} bytes:"));
			File.WriteAllBytes(Path.Combine(folder, "onnx", "model.onnx"), Weights);
			Assert.That(await HuggingFaceCache.EnsureAsync(Model(), null, folder, TestEnvironment.Nowhere(), null, CancellationToken.None), Is.EqualTo(Path.GetFullPath(folder)));
		}
		finally
		{
			Directory.Delete(folder, recursive: true);
		}
	}

	// SPLADE from a mirror with its real files (532 MB): the files of the repository, then the same vectors as from the files of the tests.
	// The options are those of the call: changed during the download, they do not reach the generator.
	[Test]
	public async Task SpladeFromAMirror()
	{
		HostedTestEnvironment.Check(HostedTestEnvironment.Splade);
		var source = HostedTestEnvironment.SpladeFolder();
		var routes = HostedModels.Splade.Files.ToDictionary(file => $"/{HostedModels.SpladeRepository}/resolve/{HostedModels.SpladeRevision}/{file.Path}",
			file => (object)HuggingFaceCache.LocalPath(source, file));
		var options = new ChromaSpladeEmbeddingOptions { CacheDirectory = _cache };
		using var mirror = new MirrorServer(routes, _ => options.IncludeTokens = true);
		options.DownloadBaseUri = mirror.BaseUri;
		using var generator = await ChromaSpladeEmbeddingGenerator.CreateAsync(options);
		var embedding = (await generator.GenerateAsync(["A sour lemon"]))[0];
		var expected = await HostedGenerators.SpladeVectorAsync("A sour lemon");
		Assert.Multiple(() =>
		{
			Assert.That(mirror.Requests, Is.EqualTo(HostedModels.Splade.Files.Count));
			Assert.That(options.IncludeTokens, Is.True, "changed during the download");
			Assert.That(generator.IncludeTokens, Is.False);
			Assert.That(embedding.Labels, Is.Null);
			Assert.That(HuggingFaceCache.HasAllFiles(HostedModels.Splade, _cache), Is.True);
			Assert.That(embedding.Indices, Is.EqualTo(expected.Indices));
			Assert.That(embedding.Values, Is.EqualTo(expected.Values));
		});
	}

	private string[] Temporary() => Directory.GetFiles(_cache, "*.tmp", SearchOption.AllDirectories);

	private static string Sha256(byte[] bytes)
	{
#if NET
		var hash = SHA256.HashData(bytes);
#else
		using var sha = SHA256.Create();
		var hash = sha.ComputeHash(bytes);
#endif
		return string.Concat(hash.Select(b => b.ToString("x2", CultureInfo.InvariantCulture)));
	}
}
