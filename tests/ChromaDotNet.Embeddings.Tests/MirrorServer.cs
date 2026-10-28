using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace ChromaDotNet.Embeddings.Tests;

// A mirror on 127.0.0.1, every GET counted. With a file or bytes, every path gets the same body: the mirror of onnx.tar.gz. With routes,
// each path of a URL has its own file or bytes, and any other path gets 404: the mirror of Hugging Face. A TcpListener with HTTP/1.1 by
// hand, as HttpListener needs a URL reservation on Windows for some prefixes.
internal sealed class MirrorServer : IDisposable
{
	private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
	private readonly string? _file;
	private readonly byte[]? _body;
	private readonly IReadOnlyDictionary<string, object>? _routes;
	private readonly Action<string>? _onRequest;
	private readonly ConcurrentDictionary<string, int> _requestsByPath = new(StringComparer.Ordinal);
	private readonly CancellationTokenSource _stop = new();
	private readonly Task _loop;
	private int _requests;

	public MirrorServer(string? file = null, byte[]? body = null)
	{
		_file = file;
		_body = body;
		_listener.Start();
		BaseUri = new Uri($"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}/");
		Uri = new Uri(BaseUri, "all-MiniLM-L6-v2/onnx.tar.gz");
		_loop = Task.Run(AcceptAsync);
	}

	// Each path, like /owner/name/resolve/revision/onnx/model.onnx, with a local file (a string) or bytes. onRequest runs with the path
	// of each request, before its response: while the client waits for it.
	public MirrorServer(IReadOnlyDictionary<string, object> routes, Action<string>? onRequest = null)
		: this()
	{
		_routes = routes;
		_onRequest = onRequest;
	}

	public Uri BaseUri { get; }

	public Uri Uri { get; }

	public int Requests => Volatile.Read(ref _requests);

	public int RequestsOf(string path) => _requestsByPath.TryGetValue(path, out var count) ? count : 0;

	private async Task AcceptAsync()
	{
		while (!_stop.IsCancellationRequested)
		{
			TcpClient client;
			try
			{
				client = await _listener.AcceptTcpClientAsync();
			}
			catch (Exception e) when (e is OperationCanceledException or SocketException or ObjectDisposedException)
			{
				return;
			}
			_ = Task.Run(() => ServeAsync(client));
		}
	}

	private async Task ServeAsync(TcpClient client)
	{
		using (client)
		{
			try
			{
				var stream = client.GetStream();
				// The request: its headers, up to the empty line; the body of a GET is empty.
				var request = new StringBuilder();
				var buffer = new byte[4096];
				while (request.ToString().IndexOf("\r\n\r\n", StringComparison.Ordinal) < 0)
				{
					var read = await stream.ReadAsync(buffer, 0, buffer.Length, _stop.Token);
					if (read == 0)
						return;
					request.Append(Encoding.ASCII.GetString(buffer, 0, read));
				}
				Interlocked.Increment(ref _requests);
				var line = request.ToString();
				var path = line.Split(' ').Length > 1 ? line.Split(' ')[1] : "/";
				_requestsByPath.AddOrUpdate(path, 1, (_, count) => count + 1);
				_onRequest?.Invoke(path);

				Stream? body;
				if (_routes is null)
					body = _file is not null ? new FileStream(_file, FileMode.Open, FileAccess.Read, FileShare.Read, 81920, useAsync: true) : new MemoryStream(_body ?? []);
				else if (_routes.TryGetValue(path, out var route))
					body = route is string file ? new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read, 81920, useAsync: true) : new MemoryStream((byte[])route);
				else
					body = null;
				using (body)
				{
					var head = body is null
						? "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
						: $"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {body.Length}\r\nConnection: close\r\n\r\n";
					var headBytes = Encoding.ASCII.GetBytes(head);
					await stream.WriteAsync(headBytes, 0, headBytes.Length, _stop.Token);
					if (body is not null)
						await body.CopyToAsync(stream, 81920, _stop.Token);
				}
			}
			catch (Exception e) when (e is IOException or OperationCanceledException or SocketException)
			{
				// The client went away, or the server stops.
			}
		}
	}

	public void Dispose()
	{
		_stop.Cancel();
		_listener.Stop();
		try
		{
			_loop.Wait(TimeSpan.FromSeconds(5));
		}
		catch (AggregateException)
		{
		}
		_stop.Dispose();
	}
}
