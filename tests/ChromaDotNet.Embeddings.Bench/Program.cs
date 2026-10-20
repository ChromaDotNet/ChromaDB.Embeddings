using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using ChromaDotNet.Embeddings;

// The speed on the 135 texts of corpus.json (each padded to 256 tokens, as in Python), after a warm-up: 3 runs of one call each, in
// batches of 32. Without an argument, the default threads; with one, that count of threads. One count per run, so that the working
// set is the one of a single generator.
//   dotnet run -c Release --project tests/ChromaDotNet.Embeddings.Bench [-- 8]
var cache = Path.Combine(typeof(Program).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().First(a => a.Key == "EmbeddingsTestRoot").Value!, "model-cache");
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
