using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ChromaDotNet.Embeddings;
using ChromaDotNet.Embeddings.Server;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);
// Only on the loopback address unless configured otherwise: then a key is required.
if (string.IsNullOrEmpty(builder.Configuration["urls"]) && string.IsNullOrEmpty(builder.Configuration["ASPNETCORE_URLS"]) &&
	string.IsNullOrEmpty(builder.Configuration["Kestrel:Endpoints:Http:Url"]))
	builder.WebHost.UseUrls("http://127.0.0.1:8080");
builder.Services.Configure<EmbeddingServerOptions>(builder.Configuration.GetSection(EmbeddingServerOptions.Section));
builder.Services.AddSingleton<Models>();
builder.Services.AddSingleton<KeyCheck>();

var app = builder.Build();
app.Lifetime.ApplicationStarted.Register(() => app.Services.GetRequiredService<KeyCheck>().Start(app));

app.Use(async (context, next) =>
{
	if (context.Request.Path == "/healthz" || app.Services.GetRequiredService<KeyCheck>().Accepts(context.Request.Headers["x-chroma-token"]))
		await next(context);
	else
		await Results.Json(new ErrorResponse("The key in x-chroma-token is missing or not the one of the service."), statusCode: StatusCodes.Status401Unauthorized).ExecuteAsync(context);
});

app.MapGet("/healthz", (Models models) => Results.Json(new
{
	status = "ok",
	models = models.Enabled.ToDictionary(m => m, m => m == EmbeddingServerOptions.QwenModel ? models.Qwen.IsLoaded : models.Splade.IsLoaded),
}));

// chroma-cloud-qwen: POST {url} with {instructions, texts}, as the clients send it to https://embed.trychroma.com.
app.MapPost("/", async (HttpRequest request, Models models, CancellationToken cancellationToken) =>
{
	if (Refuse(request, models, EmbeddingServerOptions.QwenModel, ChromaQwenEmbeddingGenerator.ModelId) is { } refused)
		return refused;
	var body = await Read<DenseRequest>(request, cancellationToken);
	if (body?.Texts is not { } texts)
		return Error(StatusCodes.Status400BadRequest, "The body must be {\"instructions\": \"...\", \"texts\": [\"...\"]}.");
	return await Run(models.Qwen, async (generator, token) =>
	{
		var embeddings = await generator.Generate(texts, body.Instructions ?? string.Empty, null, token);
		return Results.Json(new DenseResponse(embeddings.Select(e => e.Vector.ToArray()).ToArray(), embeddings.Usage?.InputTokenCount ?? 0));
	}, cancellationToken);
});

// chroma-cloud-splade: POST {url}/embed_sparse with {texts, task, target, fetch_tokens}.
app.MapPost("/embed_sparse", async (HttpRequest request, Models models, CancellationToken cancellationToken) =>
{
	if (Refuse(request, models, EmbeddingServerOptions.SpladeModel, ChromaSpladeEmbeddingGenerator.ModelId) is { } refused)
		return refused;
	var body = await Read<SparseRequest>(request, cancellationToken);
	if (body?.Texts is not { } texts)
		return Error(StatusCodes.Status400BadRequest, "The body must be {\"texts\": [\"...\"], \"task\": \"\", \"target\": \"\", \"fetch_tokens\": \"false\"}.");
	var labels = body.FetchTokens is { ValueKind: JsonValueKind.True } || body.FetchTokens is { ValueKind: JsonValueKind.String } s && s.GetString() == "true";
	return await Run(models.Splade, async (generator, token) =>
	{
		var embeddings = await generator.GenerateAsync(texts, null, token);
		return Results.Json(new SparseResponse(embeddings.Select(e => new SparseVector(e.Indices, e.Values, labels ? e.Labels : null)).ToArray()));
	}, cancellationToken);
});

app.Run();

static IResult? Refuse(HttpRequest request, Models models, string name, string modelId)
{
	var asked = request.Headers["x-chroma-embedding-model"].ToString();
	if (asked.Length > 0 && !string.Equals(asked, modelId, StringComparison.Ordinal))
		return Error(StatusCodes.Status400BadRequest, $"This endpoint computes {modelId}, not {asked}.");
	return models.Enabled.Contains(name) ? null : Error(StatusCodes.Status404NotFound, $"{modelId} is not among the Embeddings:Models of the service.");
}

static async Task<T?> Read<T>(HttpRequest request, CancellationToken cancellationToken) where T : class
{
	try
	{
		return await request.ReadFromJsonAsync<T>(cancellationToken);
	}
	catch (Exception e) when (e is JsonException or InvalidOperationException)
	{
		return null;
	}
}

static async Task<IResult> Run<TGenerator>(ModelHost<TGenerator> host, Func<TGenerator, CancellationToken, Task<IResult>> work, CancellationToken cancellationToken)
	where TGenerator : class, IDisposable
{
	try
	{
		var (accepted, result) = await host.RunAsync(work, cancellationToken);
		return accepted ? result! : Error(StatusCodes.Status503ServiceUnavailable, "The model is busy and its queue is full: try again later.");
	}
	catch (ArgumentException e)
	{
		// A text too long, or a null text: the message says which.
		var message = e.ParamName is null ? e.Message : e.Message.Replace($" (Parameter '{e.ParamName}')", string.Empty, StringComparison.Ordinal);
		return Error(message.Contains("tokens", StringComparison.Ordinal) ? StatusCodes.Status413PayloadTooLarge : StatusCodes.Status400BadRequest, message);
	}
	catch (InvalidOperationException e)
	{
		return Error(StatusCodes.Status500InternalServerError, e.Message);
	}
}

static IResult Error(int status, string message) => Results.Json(new ErrorResponse(message), statusCode: status);

internal sealed record DenseRequest(string? Instructions, string[]? Texts);

// num_tokens: the tokens read, with the instructions and <|endoftext|>, as the response of Chroma Cloud has them.
internal sealed record DenseResponse(float[][] Embeddings, [property: System.Text.Json.Serialization.JsonPropertyName("num_tokens")] long NumTokens);

internal sealed record SparseRequest(string[]? Texts, string? Task, string? Target, [property: System.Text.Json.Serialization.JsonPropertyName("fetch_tokens")] JsonElement? FetchTokens);

internal sealed record SparseVector(IReadOnlyList<int> Indices, IReadOnlyList<float> Values, IReadOnlyList<string>? Labels);

internal sealed record SparseResponse(SparseVector[] Embeddings);

internal sealed record ErrorResponse(string Error);

// The key of x-chroma-token: required until the service knows it listens only on the loopback address. Listening elsewhere without a
// key stops the service, unless Embeddings:AllowNoApiKey is set.
internal sealed class KeyCheck(IOptions<EmbeddingServerOptions> options, ILogger<KeyCheck> logger)
{
	private readonly byte[]? _key = string.IsNullOrEmpty(options.Value.ApiKey) ? null : SHA256.HashData(Encoding.UTF8.GetBytes(options.Value.ApiKey));
	private volatile bool _open;

	public void Start(WebApplication app)
	{
		var addresses = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()?.Addresses ?? [];
		var loopback = addresses.Count > 0 && addresses.All(IsLoopback);
		if (_key is not null)
			ServerLog.WithKey(logger, string.Join(", ", addresses));
		else if (loopback || options.Value.AllowNoApiKey)
		{
			_open = true;
			ServerLog.WithoutKey(logger, string.Join(", ", addresses));
		}
		else
		{
			ServerLog.OpenWithoutKey(logger, string.Join(", ", addresses));
			Environment.ExitCode = 1;
			app.Lifetime.StopApplication();
		}
	}

	public bool Accepts(string? token)
	{
		if (_key is null)
			return _open;
		return token is not null && CryptographicOperations.FixedTimeEquals(_key, SHA256.HashData(Encoding.UTF8.GetBytes(token)));
	}

	internal static bool IsLoopback(string address)
	{
		var host = new Uri(address.Replace("://+", "://0.0.0.0", StringComparison.Ordinal).Replace("://*", "://0.0.0.0", StringComparison.Ordinal)).Host.Trim('[', ']');
		return host == "localhost" || (IPAddress.TryParse(host, out var ip) && IPAddress.IsLoopback(ip));
	}
}

internal static partial class ServerLog
{
	[LoggerMessage(1, LogLevel.Information, "Listening on {Addresses}; a key is required in x-chroma-token.")]
	public static partial void WithKey(ILogger logger, string addresses);

	[LoggerMessage(2, LogLevel.Information, "Listening on {Addresses} with no key.")]
	public static partial void WithoutKey(ILogger logger, string addresses);

	[LoggerMessage(3, LogLevel.Critical, "Listening on {Addresses}, not only on the loopback address, with no key: set Embeddings:ApiKey to the key the clients send in x-chroma-token, or Embeddings:AllowNoApiKey to true to serve every client that reaches the port.")]
	public static partial void OpenWithoutKey(ILogger logger, string addresses);
}

/// <summary>The entry point, also for the tests.</summary>
public partial class Program;
