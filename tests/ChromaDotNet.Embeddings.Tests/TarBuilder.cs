using System.IO.Compression;
using System.Text;

namespace ChromaDotNet.Embeddings.Tests;

// A tar.gz of regular files with any names, "../" included, in the ustar format, the same on .NET 8 and .NET Framework (which has no
// TarWriter).
internal static class TarBuilder
{
	public static void Write(string path, params (string Name, byte[] Content)[] entries)
	{
		using var file = File.Create(path);
		using var gzip = new GZipStream(file, CompressionLevel.Fastest);
		foreach (var (name, content) in entries)
		{
			var header = new byte[512];
			Put(header, 0, 100, name);
			Put(header, 100, 8, "0000644");
			Put(header, 108, 8, "0000000");
			Put(header, 116, 8, "0000000");
			Put(header, 124, 12, Convert.ToString(content.Length, 8).PadLeft(11, '0'));
			Put(header, 136, 12, "00000000000");
			header[156] = (byte)'0';
			Put(header, 257, 6, "ustar");
			Put(header, 263, 2, "00");
			// The checksum: the sum of the bytes with its own field as spaces.
			for (var i = 148; i < 156; i++)
				header[i] = (byte)' ';
			var sum = header.Sum(b => b);
			Put(header, 148, 8, Convert.ToString(sum, 8).PadLeft(6, '0') + "\0 ");
			gzip.Write(header, 0, header.Length);
			gzip.Write(content, 0, content.Length);
			var padding = (512 - content.Length % 512) % 512;
			gzip.Write(new byte[padding], 0, padding);
		}
		gzip.Write(new byte[1024], 0, 1024);
	}

	private static void Put(byte[] header, int offset, int length, string value)
	{
		var bytes = Encoding.ASCII.GetBytes(value);
		if (bytes.Length > length)
			throw new ArgumentException($"{value} is longer than {length} bytes.", nameof(value));
		Array.Copy(bytes, 0, header, offset, bytes.Length);
	}
}
