using Microsoft.Extensions.AI;
using NUnit.Framework;

namespace ChromaDotNet.Embeddings.Tests;

[TestFixture]
public class GeneratorTests
{
	[Test]
	public async Task MetadataIsChromaAllMiniLmL6V2With384Dimensions()
	{
		IEmbeddingGenerator<string, Embedding<float>> generator = await SharedGenerator.GetAsync();
		var metadata = generator.GetService<EmbeddingGeneratorMetadata>();
		Assert.That(metadata, Is.Not.Null);
		Assert.Multiple(() =>
		{
			Assert.That(metadata!.ProviderName, Is.EqualTo("chroma"));
			Assert.That(metadata.DefaultModelId, Is.EqualTo("all-MiniLM-L6-v2"));
			Assert.That(metadata.DefaultModelDimensions, Is.EqualTo(384));
			Assert.That(metadata.ProviderUri, Is.Null);
		});
	}

	[Test]
	public async Task GetServiceGivesTheGeneratorAndNothingElse()
	{
		var generator = await SharedGenerator.GetAsync();
		Assert.Multiple(() =>
		{
			Assert.That(generator.GetService(typeof(IEmbeddingGenerator<string, Embedding<float>>)), Is.SameAs(generator));
			Assert.That(generator.GetService(typeof(ChromaDefaultEmbeddingGenerator)), Is.SameAs(generator));
			Assert.That(generator.GetService(typeof(IDisposable)), Is.SameAs(generator));
			Assert.That(generator.GetService(typeof(string)), Is.Null);
			Assert.That(generator.GetService(typeof(EmbeddingGeneratorMetadata), serviceKey: "other"), Is.Null);
			Assert.That(generator.GetService(typeof(ChromaDefaultEmbeddingGenerator), serviceKey: "other"), Is.Null);
			Assert.That(() => generator.GetService(null!), Throws.ArgumentNullException);
		});
	}

	[Test]
	public void ConstantsAreThoseOfPython()
	{
		Assert.Multiple(() =>
		{
			Assert.That(ChromaDefaultEmbeddingGenerator.EmbeddingFunctionName, Is.EqualTo("default"));
			Assert.That(ChromaDefaultEmbeddingGenerator.ModelId, Is.EqualTo("all-MiniLM-L6-v2"));
			Assert.That(ChromaDefaultEmbeddingGenerator.Dimensions, Is.EqualTo(384));
			Assert.That(ChromaDefaultEmbeddingGenerator.MaxTokens, Is.EqualTo(256));
			Assert.That(ChromaDefaultEmbeddingGenerator.DefaultSpace, Is.EqualTo("l2"));
		});
	}

	[Test]
	public async Task EmbeddingsHaveTheModelUsageAndUnitNorm()
	{
		var generator = await SharedGenerator.GetAsync();
		string[] texts = ["Hello, World!", "", "Una mela rossa"];
		var embeddings = await generator.GenerateAsync(texts);
		Assert.That(embeddings, Has.Count.EqualTo(3));
		Assert.Multiple(() =>
		{
			foreach (var embedding in embeddings)
			{
				Assert.That(embedding.ModelId, Is.EqualTo("all-MiniLM-L6-v2"));
				Assert.That(embedding.CreatedAt, Is.Not.Null);
				Assert.That(embedding.Dimensions, Is.EqualTo(384));
				var norm = Math.Sqrt(embedding.Vector.ToArray().Sum(x => (double)x * x));
				Assert.That(norm, Is.EqualTo(1).Within(1e-5));
			}
			var expected = texts.Sum(t => Math.Min(generator.Tokenizer.CountTokens(t), 256));
			Assert.That(embeddings.Usage?.InputTokenCount, Is.EqualTo(expected));
			Assert.That(embeddings.Usage?.TotalTokenCount, Is.EqualTo(expected));
		});
	}

	[Test]
	public async Task NoTextsGiveNoEmbeddings()
	{
		var generator = await SharedGenerator.GetAsync();
		var embeddings = await generator.GenerateAsync(Array.Empty<string>());
		Assert.That(embeddings, Is.Empty);
	}

	[Test]
	public async Task GenerateVectorAsyncOfMicrosoftExtensionsAiWorks()
	{
		IEmbeddingGenerator<string, Embedding<float>> generator = await SharedGenerator.GetAsync();
		var vector = await generator.GenerateVectorAsync("The cat sat on the mat");
		var python = Reference.PythonVectors[Array.IndexOf(Reference.Texts, "The cat sat on the mat")];
		Assert.That(vector.ToArray(), Is.EqualTo(python).Within(1e-6f));
	}

	[Test]
	public async Task OtherDimensionsAndNullTextsAreRejected()
	{
		var generator = await SharedGenerator.GetAsync();
		Assert.Multiple(() =>
		{
			Assert.That(() => generator.GenerateAsync(["a"], new EmbeddingGenerationOptions { Dimensions = 128 }),
				Throws.ArgumentException.With.Message.Contains("384"));
			Assert.That(() => generator.GenerateAsync(["a", null!]), Throws.ArgumentException.With.Message.Contains("at 1"));
			Assert.That(() => generator.GenerateAsync(null!), Throws.ArgumentNullException);
		});
		var same = await generator.GenerateAsync(["a"], new EmbeddingGenerationOptions { Dimensions = 384 });
		Assert.That(same[0].Dimensions, Is.EqualTo(384));
	}

	[Test]
	public async Task CancellationStopsBeforeTheNextBatch()
	{
		var generator = await SharedGenerator.GetAsync();
		using var cancelled = new CancellationTokenSource();
		cancelled.Cancel();
		Assert.That(() => generator.GenerateAsync(["a"], cancellationToken: cancelled.Token), Throws.InstanceOf<OperationCanceledException>());
	}

	// The runtime of the run, for the reports: on .NET Framework the native libraries are those of win-x64, in a 64-bit process.
	[Test]
	public async Task RunsInA64BitProcess()
	{
		var generator = await SharedGenerator.GetAsync();
		TestContext.Out.WriteLine($"runtime {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}, 64-bit process: {Environment.Is64BitProcess}, threads {generator.IntraOpNumThreads}");
		Assert.That(Environment.Is64BitProcess, Is.True);
	}

	[Test]
	public async Task DefaultThreadsAreHalfTheProcessors()
	{
		var generator = await SharedGenerator.GetAsync();
		Assert.That(generator.IntraOpNumThreads, Is.EqualTo(Math.Max(1, Environment.ProcessorCount / 2)));
	}

	// A generator of its own on the extracted files of the shared cache: after Dispose, a call fails, and Dispose again does nothing.
	[Test]
	public async Task DisposedGeneratorThrows()
	{
		var shared = await SharedGenerator.GetAsync();
		var generator = await ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { ModelDirectory = shared.ModelDirectory, IntraOpNumThreads = 2 });
		Assert.That(generator.IntraOpNumThreads, Is.EqualTo(2));
		generator.Dispose();
		generator.Dispose();
		Assert.That(() => generator.GenerateAsync(["a"]), Throws.InstanceOf<ObjectDisposedException>());
	}

	[Test]
	public void ZeroThreadsAreRejected()
	{
		var options = new ChromaDefaultEmbeddingOptions { IntraOpNumThreads = 0, CacheDirectory = Path.Combine(TestEnvironment.Root, "never-created") };
		Assert.That(() => ChromaDefaultEmbeddingGenerator.CreateAsync(options), Throws.InstanceOf<ArgumentOutOfRangeException>());
		Assert.That(Directory.Exists(options.CacheDirectory), Is.False);
	}
}
