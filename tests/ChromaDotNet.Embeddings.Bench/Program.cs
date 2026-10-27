using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using ChromaDotNet.Embeddings;

// The speed on the 135 texts of corpus.json (each padded to 256 tokens, as in Python), after a warm-up: 3 runs of one call each, in
// batches of 32. Without an argument, the default threads; with one, that count of threads. One count per run, so that the working
// set is the one of a single generator.
//   dotnet run -c Release --project tests/ChromaDotNet.Embeddings.Bench [-- 8]
// The hosted models, with the files that build/fetch-hosted-models.sh puts in artifacts/hosted-models, or in
// CHROMADOTNET_EMBEDDINGS_HOSTED_CACHE: the load, then one text at a time, on texts of the recording of Chroma Cloud. The 12 queries 3
// times, then a document of about 100 tokens 5 times, one of about 660 tokens 3 times and, for Qwen, one of about 3,300 tokens once:
// the median of each, with the tokens read. One model per run.
//   dotnet run -c Release --project tests/ChromaDotNet.Embeddings.Bench -- qwen-fp16|qwen-fp32|splade [8]
if (args.Length > 0 && args[0] is "qwen-fp16" or "qwen-fp32" or "splade")
{
	if (args.Length > 2)
	{
		Console.Error.WriteLine("One model and one count of threads at most.");
		return;
	}
	await HostedAsync(args[0], args.Length == 2 ? int.Parse(args[1], System.Globalization.CultureInfo.InvariantCulture) : null);
	return;
}

var cache =Path.Combine(typeof(Program).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().First(a => a.Key == "EmbeddingsTestRoot").Value!, "model-cache");
var texts = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "corpus.json"))).RootElement
	.EnumerateArray().Select(e => e.GetProperty("text").GetString()!).ToArray();

if (args.Length > 1)
{
	Console.Error.WriteLine("One count of threads at most.");
	return;
}
int? threads = args.Length == 1 ? int.Parse(args[0], System.Globalization.CultureInfo.InvariantCulture) : null;
var before = Process.GetCurrentProcess().WorkingSet64;
var stopwatch = Stopwatch.StartNew();
using var generator = await ChromaDefaultEmbeddingGenerator.CreateAsync(new ChromaDefaultEmbeddingOptions { CacheDirectory = cache, IntraOpNumThreads = threads });
var loadMs = stopwatch.ElapsedMilliseconds;
var first = await generator.GenerateAsync(texts);

// The largest difference from the vectors of Python, as a check of this build (NativeAOT included).
var python = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "python.json"))).RootElement.GetProperty("batch32")
	.EnumerateArray().Select(row => row.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray();
var maxDiff = first.Select((e, i) => e.Vector.ToArray().Select((x, j) => Math.Abs(x - python[i][j])).Max()).Max();
stopwatch.Restart();
const int Runs = 3;
for (var r = 0; r < Runs; r++)
	await generator.GenerateAsync(texts);
var perSecond = texts.Length * Runs / stopwatch.Elapsed.TotalSeconds;
var process = Process.GetCurrentProcess();
Console.WriteLine($"threads {threads?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? $"default ({Math.Max(1, Environment.ProcessorCount / 2)})"}: " +
	$"load {loadMs} ms, max abs diff from Python {maxDiff:E3}, {perSecond:F0} texts/s, working set {process.WorkingSet64 / 1024 / 1024} MB (before the load {before / 1024 / 1024} MB, peak {process.PeakWorkingSet64 / 1024 / 1024} MB)");

static async Task HostedAsync(string model, int? threads)
{
	var root = Environment.GetEnvironmentVariable("CHROMADOTNET_EMBEDDINGS_HOSTED_CACHE") is { Length: > 0 } variable ? variable
		: typeof(Program).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().First(a => a.Key == "EmbeddingsHostedCache").Value!;
	var cloud = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "cloud-v2.json"))).RootElement;
	var documents = cloud.GetProperty("qwen-documents").GetProperty("request").GetProperty("texts").EnumerateArray().Select(e => e.GetString()!).ToArray();
	var queries = cloud.GetProperty("qwen-queries").GetProperty("request").GetProperty("texts").EnumerateArray().Select(e => e.GetString()!).ToArray();
	var before = Process.GetCurrentProcess().WorkingSet64;
	var watch = Stopwatch.StartNew();
	IDisposable generator;
	Func<string, Task<long>> document;
	Func<string, Task<long>> query;
	if (model == "splade")
	{
		var splade = await ChromaSpladeEmbeddingGenerator.CreateAsync(new ChromaSpladeEmbeddingOptions { ModelDirectory = Path.Combine(root, "Splade_PP_en_v1"), IntraOpNumThreads = threads });
		generator = splade;
		document = query = async text => (await splade.GenerateAsync([text])).Usage!.InputTokenCount!.Value;
	}
	else
	{
		var variant = model == "qwen-fp16" ? ChromaQwenVariant.Float16 : ChromaQwenVariant.Float32;
		var folder = Path.Combine(root, "Qwen3-Embedding-0.6B", variant == ChromaQwenVariant.Float16 ? "fp16" : "fp32");
		var qwen = await ChromaQwenEmbeddingGenerator.CreateAsync(new ChromaQwenEmbeddingOptions { Variant = variant, ModelDirectory = folder, IntraOpNumThreads = threads });
		generator = qwen;
		document = async text => (await qwen.GenerateAsync([text])).Usage!.InputTokenCount!.Value;
		query = async text => (await qwen.Queries.GenerateAsync([text])).Usage!.InputTokenCount!.Value;
	}
	var loadMs = watch.ElapsedMilliseconds;
	var loaded = Process.GetCurrentProcess().WorkingSet64;
	using (generator)
	{
		watch.Restart();
		await query(queries[0]);
		var firstMs = watch.ElapsedMilliseconds;
		var line = $"{model}, threads {threads?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? $"default ({Math.Max(1, Environment.ProcessorCount / 2)})"}: " +
			$"load {loadMs} ms, first query {firstMs} ms, queries of {await MedianAsync(query, queries, 3)}, " +
			$"documents of {await MedianAsync(document, [documents[4]], 5)}, of {await MedianAsync(document, [documents[32]], 3)}";
		if (model != "splade")
			line += $", of {await MedianAsync(document, [documents[33]], 1)}";
		var process = Process.GetCurrentProcess();
		Console.WriteLine(line + $", working set {loaded / 1024 / 1024} MB after the load (before {before / 1024 / 1024} MB), peak {process.PeakWorkingSet64 / 1024 / 1024} MB");
	}
}

// The median time of texts computed one at a time, after the range of their tokens.
static async Task<string> MedianAsync(Func<string, Task<long>> embed, string[] texts, int repeat)
{
	var times = new List<double>();
	var tokens = new List<long>();
	for (var r = 0; r < repeat; r++)
	{
		foreach (var text in texts)
		{
			var watch = Stopwatch.StartNew();
			tokens.Add(await embed(text));
			times.Add(watch.Elapsed.TotalMilliseconds);
		}
	}
	times.Sort();
	var median = times.Count % 2 == 1 ? times[times.Count / 2] : (times[times.Count / 2 - 1] + times[times.Count / 2]) / 2;
	return $"{(tokens.Min() == tokens.Max() ? $"{tokens.Min()}" : $"{tokens.Min()}-{tokens.Max()}")} tokens {median:F0} ms";
}
