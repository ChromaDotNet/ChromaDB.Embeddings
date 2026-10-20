#if !NET
using System.Net.Http;
#endif

namespace ChromaDotNet.Embeddings;

// The calls that .NET Framework does not have, the same on both targets.
internal static class Compat
{
	public static void ThrowIfDisposed(bool disposed, object instance)
	{
#if NET
		ObjectDisposedException.ThrowIf(disposed, instance);
#else
		if (disposed)
			throw new ObjectDisposedException(instance.GetType().FullName);
#endif
	}

#if NET
	public static ValueTask<int> ReadAsync(Stream stream, byte[] buffer, CancellationToken cancellationToken) =>
		stream.ReadAsync(buffer.AsMemory(), cancellationToken);

	public static ValueTask WriteAsync(Stream stream, byte[] buffer, int count, CancellationToken cancellationToken) =>
		stream.WriteAsync(buffer.AsMemory(0, count), cancellationToken);
#else
	public static Task<int> ReadAsync(Stream stream, byte[] buffer, CancellationToken cancellationToken) =>
		stream.ReadAsync(buffer, 0, buffer.Length, cancellationToken);

	public static Task WriteAsync(Stream stream, byte[] buffer, int count, CancellationToken cancellationToken) =>
		stream.WriteAsync(buffer, 0, count, cancellationToken);
#endif

	public static T NotNull<T>(T? value, string name)
		where T : class => value ?? throw new ArgumentNullException(name);

	public static string ToHex(byte[] bytes)
	{
		var chars = new char[bytes.Length * 2];
		for (var i = 0; i < bytes.Length; i++)
		{
			chars[i * 2] = "0123456789abcdef"[bytes[i] >> 4];
			chars[i * 2 + 1] = "0123456789abcdef"[bytes[i] & 0xF];
		}
		return new string(chars);
	}

	// A rename onto a file that may be there.
	public static void MoveOverwrite(string source, string destination)
	{
#if NET
		File.Move(source, destination, overwrite: true);
#else
		if (File.Exists(destination))
			File.Replace(source, destination, destinationBackupFileName: null);
		else
			File.Move(source, destination);
#endif
	}

#if NET
	public static bool IsWindows => OperatingSystem.IsWindows();

	public static bool IsMacOS => OperatingSystem.IsMacOS();
#else
	// The net462 build runs on .NET Framework, on Windows.
	public static bool IsWindows => true;

	public static bool IsMacOS => false;
#endif

	public static Task<Stream> ReadAsStreamAsync(HttpContent content, CancellationToken cancellationToken)
	{
#if NET
		return content.ReadAsStreamAsync(cancellationToken);
#else
		cancellationToken.ThrowIfCancellationRequested();
		return content.ReadAsStreamAsync();
#endif
	}
}
