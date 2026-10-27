using Microsoft.Extensions.AI;
using NUnit.Framework;

namespace ChromaDotNet.Embeddings.Tests;

// Qwen3-Embedding-0.6B against the vectors Chroma Cloud recorded (TestData/hosted/cloud-v1.json and cloud-v2.json), and against
// onnxruntime 1.24.4 in Python on the same ONNX files (python-onnx.json).
[TestFixture(ChromaQwenVariant.Float16)]
[TestFixture(ChromaQwenVariant.Float32)]
public class QwenTests
{
	// Measured against Chroma Cloud, in Python with the same files: 1 - cosine at most 3.0e-6 in float32 and 3.8e-6 in float16, a
	// difference of a component at most 4.8e-4; Chroma Cloud against itself, between a batch and a text alone, 3.1e-6 and 3.2e-4.
	private const double CloudCosineTolerance = 1e-5;
	private const float CloudComponentTolerance = 1e-3f;

	private readonly ChromaQwenVariant _variant;

	public QwenTests(ChromaQwenVariant variant) => _variant = variant;

	private string PythonName => _variant == ChromaQwenVariant.Float16 ? "qwen-fp16" : "qwen-fp32";

	[Test]
	public async Task VectorsAreThoseOfCloud()
	{
		var generator = await HostedGenerators.QwenAsync(_variant);
		var worstCosine = 0d;
		var worstComponent = 0f;
		var count = 0;
		var failures = new List<string>();
		foreach (var cloudCase in HostedReference.Cases.Where(c => c.Dense is not null))
		{
			// The service got the instruction in the request: the queries have the one of nl_to_code, the documents none.
			var query = cloudCase.Instruction.Length > 0;
			if (query)
				Assert.That(cloudCase.Instruction, Is.EqualTo(generator.QueryInstruction), cloudCase.Id);
			for (var i = 0; i < cloudCase.Texts.Length; i++)
			{
				var local = await HostedGenerators.QwenVectorAsync(_variant, cloudCase.Texts[i], query);
				var cosine = 1 - Vectors.Cosine(local, cloudCase.Dense![i]);
				var component = Vectors.MaxAbsDiff(local, cloudCase.Dense[i]);
				worstCosine = Math.Max(worstCosine, cosine);
				worstComponent = Math.Max(worstComponent, component);
				count++;
				if (cosine > CloudCosineTolerance || component > CloudComponentTolerance)
					failures.Add($"{cloudCase.Id}[{i}]: 1 - cosine {cosine:E2}, max abs diff {component:E2}");
			}
		}
		TestContext.Out.WriteLine($"{_variant}: {count} vectors of Chroma Cloud, 1 - cosine at most {worstCosine:E2}, max abs diff {worstComponent:E2}");
		Assert.That(failures, Is.Empty);
	}

	// The 12 queries of cloud-v2.json against its 34 documents, by cosine: the same five first documents, in the same order.
	[Test]
	public async Task RankingIsThatOfCloud()
	{
		var documents = HostedReference.Case("qwen-documents");
		var queries = HostedReference.Case("qwen-queries");
		var localDocuments = new List<float[]>();
		foreach (var text in documents.Texts)
			localDocuments.Add(await HostedGenerators.QwenVectorAsync(_variant, text, query: false));
		var different = new List<string>();
		var worstScore = 0d;
		for (var q = 0; q < queries.Texts.Length; q++)
		{
			var local = await HostedGenerators.QwenVectorAsync(_variant, queries.Texts[q], query: true);
			var localScores = localDocuments.Select(d => Vectors.Cosine(local, d)).ToArray();
			var cloudScores = documents.Dense!.Select(d => Vectors.Cosine(queries.Dense![q], d)).ToArray();
			worstScore = Math.Max(worstScore, localScores.Zip(cloudScores, (a, b) => Math.Abs(a - b)).Max());
			var localTop = Vectors.Top(localScores, 5);
			var cloudTop = Vectors.Top(cloudScores, 5);
			if (!localTop.SequenceEqual(cloudTop))
				different.Add($"query {q}: Chroma Cloud [{string.Join(", ", cloudTop)}], local [{string.Join(", ", localTop)}]");
		}
		TestContext.Out.WriteLine($"{_variant}: same first five documents for {queries.Texts.Length - different.Count}/{queries.Texts.Length} queries, largest difference of a score {worstScore:E2}");
		Assert.That(different, Is.Empty);
	}

	// The token ids of Tokenizers.DotNet are those of tokenizers in Python, and the vectors those of onnxruntime 1.24.4 in Python.
	[Test]
	public async Task VectorsAreThoseOfOnnxRuntimeInPython()
	{
		var generator = await HostedGenerators.QwenAsync(_variant);
		var python = HostedReference.Python.GetProperty(PythonName);
		var cloud = HostedReference.Case("qwen-documents");
		var queries = HostedReference.Case("qwen-queries");
		var worst = 0f;
		var worstCosine = 0d;
		foreach (var (kind, texts, query) in new[] { ("documents", cloud.Texts, false), ("queries", queries.Texts, true) })
		{
			var expectedIds = python.GetProperty(kind).GetProperty("ids").EnumerateArray().ToArray();
			var expectedVectors = python.GetProperty(kind).GetProperty("vectors").EnumerateArray().ToArray();
			for (var i = 0; i < texts.Length; i++)
			{
				var ids = generator.Tokenizer.Encode(ChromaQwenEmbeddingGenerator.Format(query ? generator.QueryInstruction : generator.DocumentsInstruction, texts[i]), out _);
				Assert.That(ids, Is.EqualTo(expectedIds[i].EnumerateArray().Select(x => x.GetInt64()).ToArray()), $"{kind}[{i}] ids");
				var expected = expectedVectors[i].EnumerateArray().Select(x => x.GetSingle()).ToArray();
				var local = await HostedGenerators.QwenVectorAsync(_variant, texts[i], query);
				worst = Math.Max(worst, Vectors.MaxAbsDiff(local, expected));
				worstCosine = Math.Max(worstCosine, 1 - Vectors.Cosine(local, expected));
			}
		}
		TestContext.Out.WriteLine($"{_variant}: against onnxruntime in Python, max abs diff {worst:E2}, 1 - cosine at most {worstCosine:E2}");
		Assert.That(worst, Is.LessThanOrEqualTo(PythonTolerance));
	}

	[Test]
	public async Task MetadataAndQueries()
	{
		var generator = await HostedGenerators.QwenAsync(_variant);
		IEmbeddingGenerator<string, Embedding<float>> documents = generator;
		var metadata = documents.GetService<EmbeddingGeneratorMetadata>();
		Assert.Multiple(() =>
		{
			Assert.That(metadata!.ProviderName, Is.EqualTo("chroma"));
			Assert.That(metadata.DefaultModelId, Is.EqualTo("Qwen/Qwen3-Embedding-0.6B"));
			Assert.That(metadata.DefaultModelDimensions, Is.EqualTo(1024));
			Assert.That(generator.Queries.GetService<EmbeddingGeneratorMetadata>(), Is.SameAs(metadata));
			Assert.That(generator.Queries.GetService<ChromaQwenEmbeddingGenerator>(), Is.SameAs(generator));
			Assert.That(generator.Queries.GetService(typeof(IEmbeddingGenerator<string, Embedding<float>>)), Is.SameAs(generator.Queries));
			Assert.That(generator.GetService(typeof(string)), Is.Null);
			Assert.That(generator.GetService(typeof(EmbeddingGeneratorMetadata), "key"), Is.Null);
			Assert.That(generator.Variant, Is.EqualTo(_variant));
			Assert.That(generator.Task, Is.EqualTo("nl_to_code"));
			Assert.That(generator.DocumentsInstruction, Is.Empty);
			Assert.That(generator.QueryInstruction, Is.EqualTo("Given a question about coding, retrieval code or passage that can solve user's question"));
		});
		// The generator of the queries does not dispose the model.
		generator.Queries.Dispose();
		var embeddings = await generator.Queries.GenerateAsync(["a sour lemon"]);
		Assert.Multiple(() =>
		{
			Assert.That(embeddings[0].Dimensions, Is.EqualTo(1024));
			Assert.That(embeddings[0].ModelId, Is.EqualTo("Qwen/Qwen3-Embedding-0.6B"));
			Assert.That(Math.Sqrt(embeddings[0].Vector.ToArray().Sum(x => (double)x * x)), Is.EqualTo(1).Within(1e-5));
			var expected = generator.Tokenizer.Encode(ChromaQwenEmbeddingGenerator.Format(generator.QueryInstruction, "a sour lemon"), out _).Length;
			Assert.That(embeddings.Usage?.InputTokenCount, Is.EqualTo(expected));
		});
	}

	[Test]
	public async Task OtherDimensionsNullTextsAndCancellationAreRejected()
	{
		var generator = await HostedGenerators.QwenAsync(_variant);
		using var cancelled = new CancellationTokenSource();
		cancelled.Cancel();
		Assert.Multiple(() =>
		{
			Assert.That(() => generator.GenerateAsync(["a"], new EmbeddingGenerationOptions { Dimensions = 512 }), Throws.ArgumentException.With.Message.Contains("1024"));
			Assert.That(() => generator.Queries.GenerateAsync(["a", null!]), Throws.ArgumentException.With.Message.Contains("at 1"));
			Assert.That(() => generator.GenerateAsync(["a"], cancellationToken: cancelled.Token), Throws.InstanceOf<OperationCanceledException>());
		});
		Assert.That(await generator.GenerateAsync(Array.Empty<string>()), Is.Empty);
	}

	// A text of more than 32,768 tokens is truncated as tokenizers truncates one sequence: the first tokens, then <|endoftext|>.
	[Test]
	public async Task LongTextsAreTruncatedAtTheEndOfText()
	{
		var generator = await HostedGenerators.QwenAsync(_variant);
		var text = string.Concat(Enumerable.Repeat("lemon ", 40_000));
		var ids = generator.Tokenizer.Encode(text, out var tokens);
		Assert.Multiple(() =>
		{
			Assert.That(tokens, Is.GreaterThan(ChromaQwenEmbeddingGenerator.MaxTokens));
			Assert.That(ids, Has.Length.EqualTo(ChromaQwenEmbeddingGenerator.MaxTokens));
			Assert.That(generator.Tokenizer.EndId, Is.EqualTo(151643), "<|endoftext|>");
			Assert.That(ids[ids.Length - 1], Is.EqualTo(151643));
			Assert.That(generator.Tokenizer.Encode("lemon", out _).Last(), Is.EqualTo(151643));
		});
	}

	private float PythonTolerance => PythonTolerances[_variant];

	// On the machine that wrote python-onnx.json the vectors are those of Python bit for bit, in both variants. Another processor runs
	// other kernels of ONNX Runtime: the tolerance of build/python-hosted-reference.py.
	private static readonly Dictionary<ChromaQwenVariant, float> PythonTolerances = new()
	{
		[ChromaQwenVariant.Float16] = 1e-4f,
		[ChromaQwenVariant.Float32] = 1e-4f,
	};
}

// The instructions and the format of the queries, without the model.
[TestFixture]
public class QwenInstructionTests
{
	[Test]
	public void QueriesHaveTheFormatOfTheService()
	{
		Assert.Multiple(() =>
		{
			// No space after "Query:": the service builds the text so, and a space changes the vector (1 - cosine 7e-3).
			Assert.That(ChromaQwenEmbeddingGenerator.Format("Find code", "sum two numbers"), Is.EqualTo("Instruct: Find code\nQuery:sum two numbers"));
			Assert.That(ChromaQwenEmbeddingGenerator.Format(string.Empty, "sum two numbers"), Is.EqualTo("sum two numbers"));
		});
	}

	[Test]
	public void InstructionsAreChosenAsInPython()
	{
		var custom = new Dictionary<string, ChromaQwenInstructions> { ["mine"] = new("doc", "query") };
		static string Pair(ChromaQwenInstructions instructions) => instructions.Documents + "|" + instructions.Query;
		Assert.Multiple(() =>
		{
			Assert.That(Pair(ChromaQwenEmbeddingGenerator.Resolve("nl_to_code", ChromaQwenEmbeddingGenerator.DefaultInstructions)),
				Is.EqualTo("|Given a question about coding, retrieval code or passage that can solve user's question"));
			Assert.That(Pair(ChromaQwenEmbeddingGenerator.Resolve(null, ChromaQwenEmbeddingGenerator.DefaultInstructions)), Is.EqualTo("|"));
			Assert.That(Pair(ChromaQwenEmbeddingGenerator.Resolve(string.Empty, ChromaQwenEmbeddingGenerator.DefaultInstructions)), Is.EqualTo("|"));
			Assert.That(Pair(ChromaQwenEmbeddingGenerator.Resolve("other", ChromaQwenEmbeddingGenerator.DefaultInstructions)), Is.EqualTo("|"));
			Assert.That(Pair(ChromaQwenEmbeddingGenerator.Resolve("mine", custom)), Is.EqualTo("doc|query"));
			Assert.That(Pair(ChromaQwenEmbeddingGenerator.Resolve("nl_to_code", custom)), Is.EqualTo("|"));
			Assert.That(ChromaQwenEmbeddingGenerator.DefaultInstructions, Is.Not.InstanceOf<Dictionary<string, ChromaQwenInstructions>>());
		});
	}

	[Test]
	public void ConstantsAreThoseOfPython()
	{
		Assert.Multiple(() =>
		{
			Assert.That(ChromaQwenEmbeddingGenerator.EmbeddingFunctionName, Is.EqualTo("chroma-cloud-qwen"));
			Assert.That(ChromaQwenEmbeddingGenerator.ModelId, Is.EqualTo("Qwen/Qwen3-Embedding-0.6B"));
			Assert.That(ChromaQwenEmbeddingGenerator.DefaultSpace, Is.EqualTo("cosine"));
			Assert.That(ChromaQwenEmbeddingGenerator.DefaultTask, Is.EqualTo("nl_to_code"));
			Assert.That(new ChromaQwenEmbeddingOptions().Variant, Is.EqualTo(ChromaQwenVariant.Float16));
		});
	}

	[Test]
	public void ZeroThreadsAndAnUnknownVariantAreRejected()
	{
		Assert.Multiple(() =>
		{
			Assert.That(() => ChromaQwenEmbeddingGenerator.CreateAsync(new ChromaQwenEmbeddingOptions { IntraOpNumThreads = 0 }), Throws.InstanceOf<ArgumentOutOfRangeException>());
			Assert.That(() => ChromaQwenEmbeddingGenerator.CreateAsync(new ChromaQwenEmbeddingOptions { Variant = (ChromaQwenVariant)7 }), Throws.InstanceOf<ArgumentOutOfRangeException>());
			Assert.That(() => ChromaQwenEmbeddingGenerator.CreateAsync(new ChromaQwenEmbeddingOptions { Instructions = new Dictionary<string, ChromaQwenInstructions> { ["x"] = null! }, ModelDirectory = TestEnvironment.Root }),
				Throws.InstanceOf<InvalidOperationException>().Or.InstanceOf<ArgumentException>());
		});
	}
}
