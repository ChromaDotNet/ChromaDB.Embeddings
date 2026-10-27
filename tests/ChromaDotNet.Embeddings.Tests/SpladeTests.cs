using Microsoft.Extensions.AI;
using NUnit.Framework;

namespace ChromaDotNet.Embeddings.Tests;

// SPLADE (prithivida/Splade_PP_en_v1) against the vectors Chroma Cloud recorded (TestData/hosted/cloud-v1.json and cloud-v2.json), and
// against onnxruntime 1.24.4 in Python on the same ONNX file (python-onnx.json).
[TestFixture]
public class SpladeTests
{
	// Measured against Chroma Cloud, in Python with the same file: the service computes in float16, so it has 0 to 3 more terms per text,
	// with values up to 0.016, and the values of the terms of both differ by up to 0.045; the terms of the local vector are always among
	// those of the service. Chroma Cloud against itself, between a batch and a text alone: values up to 0.010 apart.
	private const float CloudOnlyTolerance = 0.02f;
	private const float CloudValueTolerance = 0.06f;

	// On the machine that wrote python-onnx.json the vectors are those of Python bit for bit. Another processor runs other kernels of
	// ONNX Runtime: the tolerance of build/python-hosted-reference.py, and a value near zero may be above zero on one side only.
	private const float PythonTolerance = 1e-3f;

	[Test]
	public async Task VectorsAreThoseOfCloud()
	{
		await HostedGenerators.SpladeAsync();
		var worstValue = 0f;
		var worstCloudOnly = 0f;
		var count = 0;
		var failures = new List<string>();
		foreach (var cloudCase in HostedReference.Cases.Where(c => c.SparseVectors is not null))
		{
			for (var i = 0; i < cloudCase.Texts.Length; i++)
			{
				var local = Vectors.ToDictionary(await HostedGenerators.SpladeVectorAsync(cloudCase.Texts[i]));
				var cloud = cloudCase.SparseVectors![i];
				count++;
				var localOnly = local.Keys.Where(index => !cloud.ContainsKey(index)).ToArray();
				var cloudOnly = cloud.Where(pair => !local.ContainsKey(pair.Key)).Select(pair => pair.Value).DefaultIfEmpty(0).Max();
				var value = local.Where(pair => cloud.ContainsKey(pair.Key)).Select(pair => Math.Abs(pair.Value - cloud[pair.Key])).DefaultIfEmpty(0).Max();
				worstValue = Math.Max(worstValue, value);
				worstCloudOnly = Math.Max(worstCloudOnly, cloudOnly);
				if (localOnly.Length > 0 || cloudOnly > CloudOnlyTolerance || value > CloudValueTolerance)
					failures.Add($"{cloudCase.Id}[{i}]: terms only here [{string.Join(", ", localOnly)}], largest term only on Chroma Cloud {cloudOnly:E2}, largest difference {value:E2}");
			}
		}
		TestContext.Out.WriteLine($"{count} vectors of Chroma Cloud: largest term only on Chroma Cloud {worstCloudOnly:E2}, largest difference of a value {worstValue:E2}");
		Assert.That(failures, Is.Empty);
	}

	// The 12 queries of cloud-v2.json against its 34 documents, by dot product: the same first document, and the same five first
	// documents. Their order can differ where Chroma Cloud has two documents closer than its own noise: in query 2, documents 8 and 0,
	// 3.9565 and 3.9520 on Chroma Cloud, whose scores change by up to 0.012 between a batch and a text alone.
	[Test]
	public async Task RankingIsThatOfCloud()
	{
		var documents = HostedReference.Case("splade-documents");
		var queries = HostedReference.Case("splade-queries");
		var localDocuments = new List<Dictionary<int, float>>();
		foreach (var text in documents.Texts)
			localDocuments.Add(Vectors.ToDictionary(await HostedGenerators.SpladeVectorAsync(text)));
		var different = new List<string>();
		var worstScore = 0d;
		for (var q = 0; q < queries.Texts.Length; q++)
		{
			var local = Vectors.ToDictionary(await HostedGenerators.SpladeVectorAsync(queries.Texts[q]));
			var localScores = localDocuments.Select(d => Vectors.Dot(local, d)).ToArray();
			var cloudScores = documents.SparseVectors!.Select(d => Vectors.Dot(queries.SparseVectors![q], d)).ToArray();
			worstScore = Math.Max(worstScore, localScores.Zip(cloudScores, (a, b) => Math.Abs(a - b)).Max());
			var localTop = Vectors.Top(localScores, 5);
			var cloudTop = Vectors.Top(cloudScores, 5);
			if (localTop[0] != cloudTop[0] || !localTop.OrderBy(i => i).SequenceEqual(cloudTop.OrderBy(i => i)))
				different.Add($"query {q}: Chroma Cloud [{string.Join(", ", cloudTop)}], local [{string.Join(", ", localTop)}]");
		}
		TestContext.Out.WriteLine($"same first document and five first documents for {queries.Texts.Length - different.Count}/{queries.Texts.Length} queries, largest difference of a score {worstScore:E2}");
		Assert.That(different, Is.Empty);
	}

	// The token ids of Tokenizers.DotNet are those of tokenizers in Python, and the vectors those of onnxruntime 1.24.4 in Python.
	[Test]
	public async Task VectorsAreThoseOfOnnxRuntimeInPython()
	{
		var generator = await HostedGenerators.SpladeAsync();
		var python = HostedReference.Python.GetProperty("splade");
		var worst = 0f;
		var different = 0;
		foreach (var (kind, texts) in new[] { ("documents", HostedReference.Case("splade-documents").Texts), ("queries", HostedReference.Case("splade-queries").Texts) })
		{
			var expectedIds = python.GetProperty(kind).GetProperty("ids").EnumerateArray().ToArray();
			var expectedIndices = python.GetProperty(kind).GetProperty("indices").EnumerateArray().ToArray();
			var expectedValues = python.GetProperty(kind).GetProperty("values").EnumerateArray().ToArray();
			for (var i = 0; i < texts.Length; i++)
			{
				var ids = generator.Tokenizer.Encode(texts[i], out _);
				Assert.That(ids, Is.EqualTo(expectedIds[i].EnumerateArray().Select(x => x.GetInt64()).ToArray()), $"{kind}[{i}] ids");
				var local = Vectors.ToDictionary(await HostedGenerators.SpladeVectorAsync(texts[i]));
				var expected = expectedIndices[i].EnumerateArray().Select(x => x.GetInt32())
					.Zip(expectedValues[i].EnumerateArray().Select(x => x.GetSingle()), (index, value) => (index, value))
					.ToDictionary(pair => pair.index, pair => pair.value);
				different += local.Keys.Count(index => !expected.ContainsKey(index)) + expected.Keys.Count(index => !local.ContainsKey(index));
				// The indices of either side, a missing one as zero.
				foreach (var index in local.Keys.Union(expected.Keys))
					worst = Math.Max(worst, Math.Abs(At(local, index) - At(expected, index)));
			}
		}
		TestContext.Out.WriteLine($"against onnxruntime in Python: {different} indices on one side only, largest difference of a value {worst:E2}");
		Assert.That(worst, Is.LessThanOrEqualTo(PythonTolerance));

		static float At(Dictionary<int, float> vector, int index) => vector.TryGetValue(index, out var value) ? value : 0f;
	}

	[Test]
	public async Task MetadataUsageAndDimensions()
	{
		var concrete = await HostedGenerators.SpladeAsync();
		var generator = concrete;
		var metadata = generator.GetService<EmbeddingGeneratorMetadata>();
		var embeddings = await generator.GenerateAsync(["A sour lemon", "A crunchy apple"]);
		Assert.Multiple(() =>
		{
			Assert.That(metadata!.ProviderName, Is.EqualTo("chroma"));
			Assert.That(metadata.DefaultModelId, Is.EqualTo("prithivida/Splade_PP_en_v1"));
			Assert.That(metadata.DefaultModelDimensions, Is.EqualTo(30522));
			Assert.That(generator.GetService<ChromaSpladeEmbeddingGenerator>(), Is.SameAs(generator));
			Assert.That(generator.GetService(typeof(string)), Is.Null);
			Assert.That(generator.GetService(typeof(EmbeddingGeneratorMetadata), "key"), Is.Null);
			Assert.That(concrete.IncludeTokens, Is.False);
			Assert.That(embeddings, Has.Count.EqualTo(2));
			Assert.That(embeddings[0].Dimensions, Is.EqualTo(30522));
			Assert.That(embeddings[0].ModelId, Is.EqualTo("prithivida/Splade_PP_en_v1"));
			Assert.That(embeddings[0].Labels, Is.Null);
			Assert.That(embeddings[0].Indices, Is.Ordered.Ascending);
			Assert.That(embeddings[0].Values, Has.All.GreaterThan(0));
			var tokens = concrete.Tokenizer.Encode("A sour lemon", out _).Length + concrete.Tokenizer.Encode("A crunchy apple", out _).Length;
			Assert.That(embeddings.Usage?.InputTokenCount, Is.EqualTo(tokens));
		});
	}

	// The tokens of the indices, as include_tokens of the Python client: the token of each index in the vocabulary of the tokenizer.
	[Test]
	public async Task TokensOfTheIndices()
	{
		HostedTestEnvironment.Check(HostedTestEnvironment.Splade);
		using var generator = await ChromaSpladeEmbeddingGenerator.CreateAsync(HostedTestEnvironment.SpladeOptions(includeTokens: true));
		var embedding = (await generator.GenerateAsync(["A sour lemon"]))[0];
		var without = await HostedGenerators.SpladeVectorAsync("A sour lemon");
		// [CLS] lemon [SEP]: the id of the token lemon.
		var lemon = (int)generator.Tokenizer.Encode("lemon", out _)[1];
		Assert.Multiple(() =>
		{
			Assert.That(generator.IncludeTokens, Is.True);
			Assert.That(embedding.Indices, Is.EqualTo(without.Indices));
			Assert.That(embedding.Values, Is.EqualTo(without.Values));
			Assert.That(embedding.Labels, Has.Count.EqualTo(embedding.Indices.Count));
			Assert.That(embedding.Indices, Does.Contain(lemon));
			Assert.That(embedding.Labels![embedding.Indices.ToList().IndexOf(lemon)], Is.EqualTo("lemon"));
			Assert.That(embedding.Labels, Does.Contain("sour"));
		});
	}

	[Test]
	public async Task OtherDimensionsNullTextsAndCancellationAreRejected()
	{
		var generator = await HostedGenerators.SpladeAsync();
		using var cancelled = new CancellationTokenSource();
		cancelled.Cancel();
		Assert.Multiple(() =>
		{
			Assert.That(() => generator.GenerateAsync(["a"], new EmbeddingGenerationOptions { Dimensions = 1024 }), Throws.ArgumentException.With.Message.Contains("30522"));
			Assert.That(() => generator.GenerateAsync(["a", null!]), Throws.ArgumentException.With.Message.Contains("at 1"));
			Assert.That(() => generator.GenerateAsync(["a"], cancellationToken: cancelled.Token), Throws.InstanceOf<OperationCanceledException>());
		});
		Assert.That(await generator.GenerateAsync(Array.Empty<string>()), Is.Empty);
	}

	// A text of more than 512 tokens is truncated as tokenizers truncates one sequence: [CLS], the first 510 tokens, then [SEP]. The
	// service truncates so too: its vectors of the two long texts of cloud-v2.json, of about 700 and 3,500 tokens, are the same.
	[Test]
	public async Task LongTextsAreTruncatedAtSep()
	{
		var generator = await HostedGenerators.SpladeAsync();
		var text = string.Concat(Enumerable.Repeat("lemon ", 600));
		var ids = generator.Tokenizer.Encode(text, out var tokens);
		var documents = HostedReference.Case("splade-documents");
		Assert.Multiple(() =>
		{
			Assert.That(tokens, Is.EqualTo(602));
			Assert.That(ids, Has.Length.EqualTo(ChromaSpladeEmbeddingGenerator.MaxTokens));
			Assert.That(ids[0], Is.EqualTo(101), "[CLS]");
			Assert.That(generator.Tokenizer.EndId, Is.EqualTo(102), "[SEP]");
			Assert.That(ids[ids.Length - 1], Is.EqualTo(102));
			Assert.That(documents.SparseVectors![33], Is.EqualTo(documents.SparseVectors[32]), "the service truncates the two long texts to the same tokens");
		});
		var embeddings = await generator.GenerateAsync([documents.Texts[32], documents.Texts[33]]);
		Assert.That(embeddings[1].Values, Is.EqualTo(embeddings[0].Values));
	}
}

// The sparse embedding itself, without the model.
[TestFixture]
public class ChromaSparseEmbeddingTests
{
	private static readonly int[] ExpectedIndices = [1, 5, 9];
	private static readonly float[] ExpectedValues = [0.5f, 1.5f, 2.5f];
	private static readonly string[] ExpectedLabels = ["a", "b", "c"];

	[Test]
	public void KeepsCopiesOfItsLists()
	{
		var indices = new List<int> { 1, 5, 9 };
		var values = new List<float> { 0.5f, 1.5f, 2.5f };
		var labels = new List<string> { "a", "b", "c" };
		var embedding = new ChromaSparseEmbedding(indices, values, labels, 10);
		indices[0] = 7;
		values[0] = 9f;
		labels[0] = "z";
		Assert.Multiple(() =>
		{
			Assert.That(embedding.Indices, Is.EqualTo(ExpectedIndices));
			Assert.That(embedding.Values, Is.EqualTo(ExpectedValues));
			Assert.That(embedding.Labels, Is.EqualTo(ExpectedLabels));
			Assert.That(embedding.Dimensions, Is.EqualTo(10));
			Assert.That(embedding.Indices, Is.Not.InstanceOf<int[]>(), "read-only");
			Assert.That(new ChromaSparseEmbedding([], []).Dimensions, Is.EqualTo(0));
		});
	}

	[Test]
	public void RejectsListsThatDoNotMatch()
	{
		Assert.Multiple(() =>
		{
			Assert.That(() => new ChromaSparseEmbedding([1, 2], [1f]), Throws.ArgumentException.With.Message.Contains("2 indices and 1 values"));
			Assert.That(() => new ChromaSparseEmbedding([1, 2], [1f, 2f], ["a"]), Throws.ArgumentException.With.Message.Contains("1 labels"));
			Assert.That(() => new ChromaSparseEmbedding([2, 1], [1f, 2f]), Throws.ArgumentException.With.Message.Contains("ascending"));
			Assert.That(() => new ChromaSparseEmbedding([1, 1], [1f, 2f]), Throws.ArgumentException.With.Message.Contains("ascending"));
			Assert.That(() => new ChromaSparseEmbedding([-1], [1f]), Throws.ArgumentException.With.Message.Contains("at least 0"));
			Assert.That(() => new ChromaSparseEmbedding([10], [1f], dimensions: 10), Throws.InstanceOf<ArgumentOutOfRangeException>());
			Assert.That(() => new ChromaSparseEmbedding([1], [1f], dimensions: -1), Throws.InstanceOf<ArgumentOutOfRangeException>());
			Assert.That(() => new ChromaSparseEmbedding(null!, [1f]), Throws.ArgumentNullException);
		});
	}
}
