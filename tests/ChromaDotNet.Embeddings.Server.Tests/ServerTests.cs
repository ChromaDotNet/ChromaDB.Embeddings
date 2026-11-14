using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using NUnit.Framework;

namespace ChromaDotNet.Embeddings.Server.Tests;

// The service over HTTP: the requests Chroma Cloud recorded on 25 October 2026 (cloud-v1.json and cloud-v2.json), sent as the clients
// send them, give the vectors Chroma Cloud gave, with the tolerances of the tests of ChromaDotNet.Embeddings. The models come from the
// cache that build/fetch-hosted-models.sh fills, offline: without their files the tests are skipped, or fail when
// CHROMADOTNET_EMBEDDINGS_HOSTED names them.
[TestFixture]
[NonParallelizable]
public class ServerTests
{
	private const string Key = "test-key";
	private const double CloudCosineTolerance = 1e-5;
	private const float CloudComponentTolerance = 1e-3f;
	private const float CloudValueTolerance = 0.06f;
	private const float CloudOnlyTolerance = 0.02f;

	private static readonly string[] OneTextArray = ["a"];
	private static readonly object OneTextBody = new { texts = OneTextArray };

	private Service _service = null!;

	[OneTimeSetUp]
	public void Start() => _service = new Service(new Dictionary<string, string?> { ["Embeddings:ApiKey"] = Key });

	[OneTimeTearDown]
	public void Stop() => _service.Dispose();

	[Test]
	public async Task QwenVectorsAreThoseOfCloud()
	{
		Require("qwen-fp16", HostedFiles.Qwen);
		var worstCosine = 0d;
		var worstComponent = 0f;
		var count = 0;
		foreach (var recorded in Recordings.Cases.Where(c => !c.Sparse))
		{
			var response = await _service.PostAsync("/", recorded.Request, recorded.Model);
			Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK), recorded.Name);
			var vectors = (await response.Content.ReadFromJsonAsync<JsonObject>())!["embeddings"]!.AsArray();
			Assert.That(vectors, Has.Count.EqualTo(recorded.Vectors.Length), recorded.Name);
			for (var i = 0; i < vectors.Count; i++)
			{
				var local = vectors[i]!.AsArray().Select(v => v!.GetValue<float>()).ToArray();
				var cloud = recorded.Vectors[i].EnumerateArray().Select(v => v.GetSingle()).ToArray();
				worstCosine = Math.Max(worstCosine, 1 - Cosine(local, cloud));
				worstComponent = Math.Max(worstComponent, local.Zip(cloud, (a, b) => Math.Abs(a - b)).Max());
				count++;
			}
		}
		TestContext.Out.WriteLine($"{count} vectors of Chroma Cloud through HTTP: 1 - cosine at most {worstCosine:E2}, max abs diff {worstComponent:E2}");
		Assert.That(worstCosine, Is.LessThanOrEqualTo(CloudCosineTolerance));
		Assert.That(worstComponent, Is.LessThanOrEqualTo(CloudComponentTolerance));
	}

	[Test]
	public async Task SpladeVectorsAreThoseOfCloud()
	{
		Require("splade", HostedFiles.Splade);
		var worstValue = 0f;
		var worstOnlyCloud = 0f;
		var count = 0;
		foreach (var recorded in Recordings.Cases.Where(c => c.Sparse))
		{
			var response = await _service.PostAsync("/embed_sparse", recorded.Request, recorded.Model);
			Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK), recorded.Name);
			var vectors = (await response.Content.ReadFromJsonAsync<JsonObject>())!["embeddings"]!.AsArray();
			var fetchTokens = recorded.Request.GetProperty("fetch_tokens").GetString() == "true";
			for (var i = 0; i < vectors.Count; i++)
			{
				var local = Sparse(vectors[i]!["indices"]!.AsArray(), vectors[i]!["values"]!.AsArray());
				var cloud = recorded.Vectors[i].GetProperty("indices").EnumerateArray().Zip(recorded.Vectors[i].GetProperty("values").EnumerateArray(),
					(index, value) => (index.GetInt32(), value.GetSingle())).ToDictionary(p => p.Item1, p => p.Item2);
				Assert.That(local.Keys.All(cloud.ContainsKey), recorded.Name + ": the terms of the service are among those of Chroma Cloud");
				worstValue = Math.Max(worstValue, local.Max(p => Math.Abs(p.Value - cloud[p.Key])));
				worstOnlyCloud = Math.Max(worstOnlyCloud, cloud.Where(p => !local.ContainsKey(p.Key)).Select(p => p.Value).DefaultIfEmpty(0).Max());
				var labels = vectors[i]!["labels"];
				Assert.That(labels is null, Is.EqualTo(!fetchTokens), recorded.Name + ": the labels only when fetch_tokens is \"true\"");
				if (labels is not null)
					Assert.That(labels.AsArray(), Has.Count.EqualTo(local.Count));
				count++;
			}
		}
		TestContext.Out.WriteLine($"{count} vectors of Chroma Cloud through HTTP: values up to {worstValue:E2} apart, terms only of Chroma Cloud up to {worstOnlyCloud:E2}");
		Assert.That(worstValue, Is.LessThanOrEqualTo(CloudValueTolerance));
		Assert.That(worstOnlyCloud, Is.LessThanOrEqualTo(CloudOnlyTolerance));
	}

	// A text of more than 8192 tokens is refused with 413 and the message of the package, before any text of the request is computed.
	[Test]
	public async Task QwenRefusesATextOfMoreThan8192Tokens()
	{
		Require("qwen-fp16", HostedFiles.Qwen);
		var response = await _service.PostAsync("/", JsonSerializer.SerializeToElement(new { instructions = "", texts = new[] { "short", string.Concat(Enumerable.Repeat("word ", 9000)) } }), "Qwen/Qwen3-Embedding-0.6B");
		Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.RequestEntityTooLarge));
		var error = (await response.Content.ReadFromJsonAsync<JsonObject>())!["error"]!.GetValue<string>();
		Assert.That(error, Does.StartWith("The text at 1 has 900").And.Contain("at most 8192 tokens").And.Not.Contain("Parameter"));
	}

	[Test]
	public async Task TheKeyIsRequired()
	{
		using var client = _service.CreateClient();
		using var missing = await client.PostAsJsonAsync("/embed_sparse", OneTextBody);
		Assert.That(missing.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized));
		using var request = new HttpRequestMessage(HttpMethod.Post, "/embed_sparse") { Content = JsonContent.Create(OneTextBody) };
		request.Headers.Add("x-chroma-token", "other-key");
		using var wrong = await client.SendAsync(request);
		Assert.That(wrong.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized));
		using var health = await client.GetAsync("/healthz");
		Assert.That(health.StatusCode, Is.EqualTo(HttpStatusCode.OK));
	}

	[Test]
	public async Task ABadRequestGetsAnError()
	{
		var noTexts = await _service.PostAsync("/", JsonSerializer.SerializeToElement(new { instructions = "" }), null);
		Assert.That(noTexts.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
		Assert.That((await noTexts.Content.ReadFromJsonAsync<JsonObject>())!["error"], Is.Not.Null);
		var otherModel = await _service.PostAsync("/embed_sparse", JsonSerializer.SerializeToElement(OneTextBody), "Qwen/Qwen3-Embedding-0.6B");
		Assert.That(otherModel.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
	}

	[Test]
	public async Task AModelNotEnabledIsNotServed()
	{
		using var service = new Service(new Dictionary<string, string?> { ["Embeddings:ApiKey"] = Key, ["Embeddings:Models"] = "splade" });
		var response = await service.PostAsync("/", JsonSerializer.SerializeToElement(new { instructions = "", texts = OneTextArray }), null);
		Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
	}

	// One request at a time per model: with no room in the queue, a request that comes while another one is computed gets 503.
	[Test]
	public async Task ARequestBeyondTheQueueGets503()
	{
		Require("splade", HostedFiles.Splade);
		using var service = new Service(new Dictionary<string, string?> { ["Embeddings:ApiKey"] = Key, ["Embeddings:QueueLimit"] = "0" });
		var body = JsonSerializer.SerializeToElement(new { texts = Enumerable.Repeat(string.Concat(Enumerable.Repeat("lemon ", 500)), 40).ToArray() });
		var responses = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => service.PostAsync("/embed_sparse", body, null)));
		var codes = responses.Select(r => r.StatusCode).ToArray();
		Assert.That(codes, Has.Some.EqualTo(HttpStatusCode.OK));
		Assert.That(codes, Has.Some.EqualTo(HttpStatusCode.ServiceUnavailable));
		Assert.That(codes, Has.All.EqualTo(HttpStatusCode.OK).Or.EqualTo(HttpStatusCode.ServiceUnavailable));
	}

	[TestCase("http://127.0.0.1:8080", true)]
	[TestCase("http://[::1]:8080", true)]
	[TestCase("http://localhost:8080", true)]
	[TestCase("http://0.0.0.0:8080", false)]
	[TestCase("http://+:8080", false)]
	[TestCase("http://*:8080", false)]
	[TestCase("http://[::]:8080", false)]
	[TestCase("http://192.168.1.5:8080", false)]
	public void OnlyTheLoopbackAddressNeedsNoKey(string address, bool loopback) =>
		Assert.That(KeyCheck.IsLoopback(address), Is.EqualTo(loopback));

	private static void Require(string name, bool present)
	{
		if (present)
			return;
		var required = Environment.GetEnvironmentVariable("CHROMADOTNET_EMBEDDINGS_HOSTED")?.Split(',').Select(s => s.Trim()).Contains(name) == true;
		var message = $"No files of {name} in {HostedFiles.Cache}: run build/fetch-hosted-models.sh.";
		if (required)
			Assert.Fail(message);
		Assert.Ignore(message);
	}

	private static Dictionary<int, float> Sparse(JsonArray indices, JsonArray values) =>
		indices.Zip(values, (i, v) => (i!.GetValue<int>(), v!.GetValue<float>())).ToDictionary(p => p.Item1, p => p.Item2);

	private static double Cosine(float[] a, float[] b)
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

	// The service in memory, offline, on the cache of the tests, with the settings given.
	private sealed class Service(IDictionary<string, string?> settings) : WebApplicationFactory<Program>
	{
		protected override void ConfigureWebHost(IWebHostBuilder builder)
		{
			builder.UseSetting("Embeddings:CacheDirectory", HostedFiles.Cache);
			builder.UseSetting("Embeddings:Offline", "true");
			foreach (var (key, value) in settings)
				builder.UseSetting(key, value);
		}

		public async Task<HttpResponseMessage> PostAsync(string path, JsonElement body, string? model)
		{
			using var client = CreateClient();
			using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = JsonContent.Create(body) };
			request.Headers.Add("x-chroma-token", Key);
			if (model is not null)
				request.Headers.Add("x-chroma-embedding-model", model);
			var response = await client.SendAsync(request);
			await response.Content.LoadIntoBufferAsync();
			return response;
		}
	}
}

internal static class HostedFiles
{
	public static string Cache { get; } = Environment.GetEnvironmentVariable("CHROMADOTNET_EMBEDDINGS_HOSTED_CACHE") ??
		typeof(HostedFiles).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), false)
			.Cast<System.Reflection.AssemblyMetadataAttribute>().First(a => a.Key == "EmbeddingsHostedCache").Value!;

	public static bool Qwen => HuggingFaceCache.HasAllFiles(HostedModels.QwenFloat16, Path.Combine(Cache, HostedModels.QwenFolder, "fp16"));

	public static bool Splade => HuggingFaceCache.HasAllFiles(HostedModels.Splade, Path.Combine(Cache, HostedModels.SpladeFolder));
}

// The requests of the recordings, as they were sent, with the model of their header and the vectors of the response.
internal sealed record Recording(string Name, string Model, bool Sparse, JsonElement Request, JsonElement[] Vectors);

internal static class Recordings
{
	public static IReadOnlyList<Recording> Cases { get; } = [.. Read("cloud-v1.json"), .. Read("cloud-v2.json")];

	private static IEnumerable<Recording> Read(string file)
	{
		var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "TestData", file)));
		foreach (var property in document.RootElement.EnumerateObject().Where(p => p.Value.ValueKind == JsonValueKind.Object))
		{
			var model = property.Value.GetProperty("model").GetString()!;
			yield return new Recording(file + ":" + property.Name, model, model == ChromaSpladeEmbeddingGenerator.ModelId,
				property.Value.GetProperty("request").Clone(), [.. property.Value.GetProperty("vectors").EnumerateArray().Select(v => v.Clone())]);
		}
	}
}
