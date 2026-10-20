using System.Net;
using System.Net.Sockets;
using System.Text;

namespace ChromaDotNet.Embeddings.Tests;

// A mirror of onnx.tar.gz on 127.0.0.1: every GET gets the same body, a file or bytes, and is counted. A TcpListener with HTTP/1.1 by
// hand, as HttpListener needs a URL reservation on Windows for some prefixes.
internal sealed class MirrorServer : IDisposable
{
	private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
	private readonly string? _file;
	private readonly byte[]? _body;
	private readonly CancellationTokenSource _stop = new();
	private readonly Task _loop;
	private int _requests;

	public MirrorServer(string? file = null, byte[]? body = null)
	{
		_file = file;
		_body = body;
		_listener.Start();
		Uri = new Uri($"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}/all-MiniLM-L6-v2/onnx.tar.gz");
		_loop = Task.Run(AcceptAsync);
	}

	public Uri Uri { get; }

	public int Requests => Volatile.Read(ref _requests);

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
				while (!request.ToString().Contains("\r\n\r\n", StringComparison.Ordinal))
				{
					var read = await stream.ReadAsync(buffer, 0, buffer.Length, _stop.Token);
					if (read == 0)
						return;
					request.Append(Encoding.ASCII.GetString(buffer, 0, read));
				}
				Interlocked.Increment(ref _requests);
				using Stream body = _file is not null ? new FileStream(_file, FileMode.Open, FileAccess.Read, FileShare.Read, 81920, useAsync: true) : new MemoryStream(_body ?? []);
				var head = $"HTTP/1.1 200 OK\r\nContent-Type: application/gzip\r\nContent-Length: {body.Length}\r\nConnection: close\r\n\r\n";
				var headBytes = Encoding.ASCII.GetBytes(head);
				await stream.WriteAsync(headBytes, 0, headBytes.Length, _stop.Token);
				await body.CopyToAsync(stream, 81920, _stop.Token);
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
