using System.Text.Json;
using NUnit.Framework;

namespace ChromaDotNet.Embeddings.Tests;

// The files of the hosted models for the tests, never downloaded by the tests:
//   CHROMADOTNET_EMBEDDINGS_HOSTED_MODELS: a folder with a folder for each repository of Hugging Face, named owner__name, in the layout
//     of the repository (onnx-community__Qwen3-Embedding-0.6B-ONNX, prithivida__Splade_PP_en_v1): used as ModelDirectory;
//   CHROMADOTNET_EMBEDDINGS_HOSTED_CACHE: the cache of the package, as build/fetch-hosted-models.sh fills it, used as CacheDirectory;
//     artifacts/hosted-models by default, where the script puts it;
//   CHROMADOTNET_EMBEDDINGS_HOSTED: the models a run must test, like qwen-fp16,splade; the others are skipped. Without it, a model is
//     tested when its files are there, and skipped otherwise;
//   CHROMADOTNET_EMBEDDINGS_HOSTED_PYTHON: the vectors of onnxruntime in Python on this machine, that build/python-hosted-reference.py
//     --output wrote: the CI computes them on each runner. Without it, the comparison on the same machine is skipped.
internal static class HostedTestEnvironment
{
	public const string QwenFloat16 = "qwen-fp16";
	public const string QwenFloat32 = "qwen-fp32";
	public const string Splade = "splade";

	private static string? ModelsRoot => Variable("CHROMADOTNET_EMBEDDINGS_HOSTED_MODELS");

	private static string? CacheRoot => Variable("CHROMADOTNET_EMBEDDINGS_HOSTED_CACHE") ?? DefaultCache;

	private static string DefaultCache { get; } = typeof(HostedTestEnvironment).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), false)
		.Cast<System.Reflection.AssemblyMetadataAttribute>().First(a => a.Key == "EmbeddingsHostedCache").Value!;

	private static string[]? Required => Variable("CHROMADOTNET_EMBEDDINGS_HOSTED")?.Split(',').Select(s => s.Trim()).Where(s => s.Length > 0).ToArray();

	public static string? PythonOfThisMachinePath => Variable("CHROMADOTNET_EMBEDDINGS_HOSTED_PYTHON");

	public static string Name(ChromaQwenVariant variant) => variant == ChromaQwenVariant.Float16 ? QwenFloat16 : QwenFloat32;

	// The options with the files of the variant, or the test is skipped (or fails, when the run must test it).
	public static ChromaQwenEmbeddingOptions Qwen(ChromaQwenVariant variant)
	{
		var model = HostedModels.Qwen(variant);
		if (ModelsRoot is { } root && HuggingFaceCache.HasAllFiles(model, Path.Combine(root, "onnx-community__Qwen3-Embedding-0.6B-ONNX")))
			return new ChromaQwenEmbeddingOptions { Variant = variant, ModelDirectory = Path.Combine(root, "onnx-community__Qwen3-Embedding-0.6B-ONNX") };
		if (CacheRoot is { } cache && HuggingFaceCache.HasAllFiles(model, Path.Combine(cache, HostedModels.QwenFolder, model.Subfolder!)))
			return new ChromaQwenEmbeddingOptions { Variant = variant, CacheDirectory = Path.Combine(cache, HostedModels.QwenFolder), DownloadBaseUri = TestEnvironment.Nowhere() };
		throw Missing(Name(variant));
	}

	public static ChromaSpladeEmbeddingOptions SpladeOptions(bool includeTokens = false)
	{
		if (ModelsRoot is { } root && HuggingFaceCache.HasAllFiles(HostedModels.Splade, Path.Combine(root, "prithivida__Splade_PP_en_v1")))
			return new ChromaSpladeEmbeddingOptions { ModelDirectory = Path.Combine(root, "prithivida__Splade_PP_en_v1"), IncludeTokens = includeTokens };
		if (CacheRoot is { } cache && HuggingFaceCache.HasAllFiles(HostedModels.Splade, Path.Combine(cache, HostedModels.SpladeFolder)))
			return new ChromaSpladeEmbeddingOptions { CacheDirectory = Path.Combine(cache, HostedModels.SpladeFolder), DownloadBaseUri = TestEnvironment.Nowhere(), IncludeTokens = includeTokens };
		throw Missing(Splade);
	}

	// The folder with the files of a variant of Qwen3-Embedding or of SPLADE, for the tests of the cache.
	public static string QwenFolder(ChromaQwenVariant variant)
	{
		var options = Qwen(variant);
		return options.ModelDirectory ?? Path.Combine(options.CacheDirectory!, HostedModels.Qwen(variant).Subfolder!);
	}

	public static string SpladeFolder()
	{
		var options = SpladeOptions();
		return options.ModelDirectory ?? options.CacheDirectory!;
	}

	// A model the run must test fails when its files are missing; another one is skipped, also when its files are there.
	public static void Check(string name)
	{
		if (Required is { } required && !required.Contains(name, StringComparer.Ordinal))
			throw new IgnoreException($"{name} is not in CHROMADOTNET_EMBEDDINGS_HOSTED.");
	}

	private static Exception Missing(string name)
	{
		var message = $"No files of {name}: set CHROMADOTNET_EMBEDDINGS_HOSTED_MODELS to a folder with the repositories of Hugging Face, or " +
			"CHROMADOTNET_EMBEDDINGS_HOSTED_CACHE to a cache filled by build/fetch-hosted-models.sh.";
		return Required is { } required && required.Contains(name, StringComparer.Ordinal) ? new AssertionException(message) : new IgnoreException(message);
	}

	private static string? Variable(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : null;
}

// One generator of each hosted model for the tests that compute vectors, and the vectors of each text, computed once: one text at a
// time, a text gives the same vector in any call.
[SetUpFixture]
public sealed class HostedGenerators
{
	private static readonly SemaphoreSlim Gate = new(1, 1);
	private static readonly Dictionary<ChromaQwenVariant, ChromaQwenEmbeddingGenerator> Qwens = [];
	private static readonly Dictionary<(ChromaQwenVariant, bool, string), float[]> QwenVectors = [];
	private static readonly Dictionary<string, ChromaSparseEmbedding> SpladeVectors = new(StringComparer.Ordinal);
	private static ChromaSpladeEmbeddingGenerator? _splade;

	internal static async Task<ChromaQwenEmbeddingGenerator> QwenAsync(ChromaQwenVariant variant)
	{
		HostedTestEnvironment.Check(HostedTestEnvironment.Name(variant));
		await Gate.WaitAsync();
		try
		{
			if (!Qwens.TryGetValue(variant, out var generator))
				Qwens[variant] = generator = await ChromaQwenEmbeddingGenerator.CreateAsync(HostedTestEnvironment.Qwen(variant));
			return generator;
		}
		finally
		{
			Gate.Release();
		}
	}

	internal static async Task<ChromaSpladeEmbeddingGenerator> SpladeAsync()
	{
		HostedTestEnvironment.Check(HostedTestEnvironment.Splade);
		await Gate.WaitAsync();
		try
		{
			return _splade ??= await ChromaSpladeEmbeddingGenerator.CreateAsync(HostedTestEnvironment.SpladeOptions());
		}
		finally
		{
			Gate.Release();
		}
	}

	// The vector of a document, or of a query with the instruction of the queries.
	internal static async Task<float[]> QwenVectorAsync(ChromaQwenVariant variant, string text, bool query)
	{
		var generator = await QwenAsync(variant);
		lock (QwenVectors)
		{
			if (QwenVectors.TryGetValue((variant, query, text), out var known))
				return known;
		}
		var vector = (await (query ? generator.Queries : generator).GenerateAsync([text]))[0].Vector.ToArray();
		lock (QwenVectors)
			QwenVectors[(variant, query, text)] = vector;
		return vector;
	}

	internal static async Task<ChromaSparseEmbedding> SpladeVectorAsync(string text)
	{
		var generator = await SpladeAsync();
		lock (SpladeVectors)
		{
			if (SpladeVectors.TryGetValue(text, out var known))
				return known;
		}
		var vector = (await generator.GenerateAsync([text]))[0];
		lock (SpladeVectors)
			SpladeVectors[text] = vector;
		return vector;
	}

	[OneTimeTearDown]
	public void TearDown()
	{
		foreach (var generator in Qwens.Values)
			generator.Dispose();
		_splade?.Dispose();
	}
}

// The recordings of Chroma Cloud (TestData/hosted/cloud-v1.json and cloud-v2.json) and the reference of Python with onnxruntime 1.24.4
// (python-onnx.json, written by build/python-hosted-reference.py), with the vectors of Python on this machine when the run has them.
internal static class HostedReference
{
	private static readonly Lazy<IReadOnlyList<CloudCase>> CasesData = new(() => [.. ReadCases("cloud-v1.json"), .. ReadCases("cloud-v2.json")]);
	private static readonly Lazy<JsonDocument> PythonData = new(() => JsonDocument.Parse(File.ReadAllBytes(DataPath("python-onnx.json"))));
	private static readonly Lazy<JsonDocument?> PythonOfThisMachineData = new(() =>
		HostedTestEnvironment.PythonOfThisMachinePath is { } path ? JsonDocument.Parse(File.ReadAllBytes(path)) : null);

	public static IReadOnlyList<CloudCase> Cases => CasesData.Value;

	public static CloudCase Case(string name) => Cases.Single(c => c.Name == name);

	public static JsonElement Python => PythonData.Value.RootElement;

	// The vectors of onnxruntime in Python on this machine, from CHROMADOTNET_EMBEDDINGS_HOSTED_PYTHON, or null.
	public static JsonElement? PythonOfThisMachine => PythonOfThisMachineData.Value?.RootElement;

	private static List<CloudCase> ReadCases(string file)
	{
		using var document = JsonDocument.Parse(File.ReadAllBytes(DataPath(file)));
		var cases = new List<CloudCase>();
		foreach (var property in document.RootElement.EnumerateObject())
		{
			if (property.Value.ValueKind != JsonValueKind.Object)
				continue;
			var request = property.Value.GetProperty("request");
			var texts = request.GetProperty("texts").EnumerateArray().Select(e => e.GetString()!).ToArray();
			var instruction = request.TryGetProperty("instructions", out var i) ? i.GetString()! : string.Empty;
			var sparse = property.Value.GetProperty("model").GetString() == ChromaSpladeEmbeddingGenerator.ModelId;
			var vectors = property.Value.GetProperty("vectors").EnumerateArray().ToArray();
			cases.Add(new CloudCase(
				file + ":" + property.Name,
				property.Name,
				texts,
				instruction,
				sparse ? null : vectors.Select(v => v.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray(),
				sparse ? vectors.Select(v => Sparse(v.GetProperty("indices"), v.GetProperty("values"))).ToArray() : null));
		}
		return cases;
	}

	public static Dictionary<int, float> Sparse(JsonElement indices, JsonElement values) =>
		indices.EnumerateArray().Zip(values.EnumerateArray(), (i, v) => (i.GetInt32(), v.GetSingle())).ToDictionary(p => p.Item1, p => p.Item2);

	public static string DataPath(string name) => Path.Combine(AppContext.BaseDirectory, "TestData", "hosted", name);
}

// A recorded request: its texts, the instruction it sent, and the vectors the service gave, dense for Qwen and sparse for SPLADE.
internal sealed class CloudCase
{
	public CloudCase(string id, string name, string[] texts, string instruction, float[][]? dense, Dictionary<int, float>[]? sparse)
	{
		Id = id;
		Name = name;
		Texts = texts;
		Instruction = instruction;
		Dense = dense;
		SparseVectors = sparse;
	}

	public string Id { get; }

	public string Name { get; }

	public string[] Texts { get; }

	public string Instruction { get; }

	public float[][]? Dense { get; }

	public Dictionary<int, float>[]? SparseVectors { get; }

	public override string ToString() => Id;
}

internal static class Vectors
{
	public static double Cosine(float[] a, float[] b)
	{
		double dot = 0, na = 0, nb = 0;
		for (var i = 0; i < a.Length; i++)
		{
			dot += (double)a[i] * b[i];
			na += (double)a[i] * a[i];
			nb += (double)b[i] * b[i];
		}
		return dot / Math.Sqrt(na * nb);
	}

	public static float MaxAbsDiff(float[] a, float[] b)
	{
		var max = 0f;
		for (var i = 0; i < a.Length; i++)
			max = Math.Max(max, Math.Abs(a[i] - b[i]));
		return max;
	}

	public static double Dot(IReadOnlyDictionary<int, float> a, IReadOnlyDictionary<int, float> b) =>
		a.Sum(pair => b.TryGetValue(pair.Key, out var v) ? (double)pair.Value * v : 0);

	public static Dictionary<int, float> ToDictionary(ChromaSparseEmbedding embedding) =>
		embedding.Indices.Zip(embedding.Values, (i, v) => (i, v)).ToDictionary(p => p.i, p => p.v);

	// The indices of the k largest scores, the first first; equal scores in the order of their indices.
	public static int[] Top(IReadOnlyList<double> scores, int k) =>
		Enumerable.Range(0, scores.Count).OrderByDescending(i => scores[i]).ThenBy(i => i).Take(k).ToArray();
}
