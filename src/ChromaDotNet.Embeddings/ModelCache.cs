using System.Diagnostics;
#if !NET
using System.Net.Http;
#endif
#if NET
using System.Formats.Tar;
#endif
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using Microsoft.Extensions.Logging;

namespace ChromaDotNet.Embeddings;

// The model files, as the Python client keeps them (chromadb 1.5.9, ONNXMiniLM_L6_V2): onnx.tar.gz in the cache folder, and the onnx/
// folder extracted from it. As in Python, the six files of onnx/ are used when they are all there, without hashing them; otherwise the
// archive is used when its SHA-256 hash matches, or downloaded. A lock file of the folder keeps a second process from downloading at
// the same time: it waits, then finds the files there. Each file is written to a temporary name and renamed when complete, so a file
// that is there is whole.
internal static class ModelCache
{
	public const string DefaultDownloadUri = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz";
	public const string ArchiveSha256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3";
	public const string ArchiveFileName = "onnx.tar.gz";
	public const string ExtractedFolderName = "onnx";
	private const string LockFileName = "onnx.tar.gz.lock";

	// The files of onnx/ that Python checks, in the order of its list.
	public static readonly string[] ModelFiles = ["config.json", "model.onnx", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"];

	// The two files this library reads, the least a ModelDirectory needs.
	private static readonly string[] RequiredFiles = ["model.onnx", "tokenizer.json"];

	private static readonly Lazy<HttpClient> Http = new(() =>
	{
		var client = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
		var version = typeof(ModelCache).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion.Split('+')[0] ?? "0";
		client.DefaultRequestHeaders.UserAgent.ParseAdd($"ChromaDotNet.Embeddings/{version}");
		return client;
	});

	// Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2", as Python: the profile of the user on Windows, $HOME elsewhere.
	public static string DefaultCacheDirectory()
	{
		var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
		if (string.IsNullOrEmpty(home))
			throw new InvalidOperationException("No home folder of the user for the model: set ChromaDefaultEmbeddingOptions.CacheDirectory or ModelDirectory.");
		return Path.Combine(home, ".cache", "chroma", "onnx_models", ChromaDefaultEmbeddingGenerator.ModelId);
	}

	// The folder with the files of the model: ModelDirectory as it is, or the onnx/ folder of the cache, downloaded and extracted when
	// its files are not all there.
	public static async Task<string> EnsureAsync(ChromaDefaultEmbeddingOptions options, CancellationToken cancellationToken)
	{
		var uri = options.DownloadUri ?? new Uri(DefaultDownloadUri);
		if (options.ModelDirectory is { } modelDirectory)
		{
			var full = Path.GetFullPath(modelDirectory);
			var missing = RequiredFiles.Where(name => !File.Exists(Path.Combine(full, name))).ToArray();
			if (missing.Length > 0)
				throw new InvalidOperationException(
					$"ChromaDefaultEmbeddingOptions.ModelDirectory is {full}, which has no {string.Join(" and ", missing)}: set it to the onnx/ folder " +
					$"extracted from {uri}, or leave it null to download the model.");
			return full;
		}

		var directory = Path.GetFullPath(options.CacheDirectory ?? DefaultCacheDirectory());
		var onnx = Path.Combine(directory, ExtractedFolderName);
		if (HasAllFiles(onnx))
			return onnx;
		try
		{
			Directory.CreateDirectory(directory);
		}
		catch (Exception e) when (e is IOException or UnauthorizedAccessException)
		{
			throw CacheError(directory, e);
		}

		using (await LockAsync(Path.Combine(directory, LockFileName), cancellationToken).ConfigureAwait(false))
		{
			// Another process may have finished while this one waited for the lock.
			if (HasAllFiles(onnx))
				return onnx;
			var archive = Path.Combine(directory, ArchiveFileName);
			if (File.Exists(archive) && await HashFileAsync(archive, cancellationToken).ConfigureAwait(false) == ArchiveSha256)
			{
				if (options.Logger is { } logger)
					Log.ArchiveFound(logger, archive);
			}
			else
			{
				await DownloadAsync(uri, archive, directory, options.Logger, cancellationToken).ConfigureAwait(false);
			}
			await ExtractAsync(archive, onnx, options.Logger, cancellationToken).ConfigureAwait(false);
		}
		return onnx;
	}

	internal static bool HasAllFiles(string onnx) => ModelFiles.All(name => File.Exists(Path.Combine(onnx, name)));

	private static async Task DownloadAsync(Uri uri, string path, string directory, ILogger? logger, CancellationToken cancellationToken)
	{
		var temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
		var stopwatch = Stopwatch.StartNew();
		if (logger is not null)
			Log.Downloading(logger, uri, path);
		try
		{
			string actualHash;
			long length;
			try
			{
				using var response = await Http.Value.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
				if (!response.IsSuccessStatusCode)
					throw new HttpRequestException($"HTTP {(int)response.StatusCode} {response.ReasonPhrase}");
				using var body = await Compat.ReadAsStreamAsync(response.Content, cancellationToken).ConfigureAwait(false);
				using var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 81920, useAsync: true);
				using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
				var buffer = new byte[81920];
				int read;
				while ((read = await Compat.ReadAsync(body, buffer, cancellationToken).ConfigureAwait(false)) > 0)
				{
					sha.AppendData(buffer, 0, read);
					await Compat.WriteAsync(file, buffer, read, cancellationToken).ConfigureAwait(false);
				}
				actualHash = Compat.ToHex(sha.GetHashAndReset());
				length = file.Length;
			}
			catch (Exception e) when (e is HttpRequestException or IOException || (e is TaskCanceledException && !cancellationToken.IsCancellationRequested))
			{
				throw new InvalidOperationException(
					$"Could not download the model {ChromaDefaultEmbeddingGenerator.ModelId} from {uri}: {e.Message} Without the network: set " +
					$"ChromaDefaultEmbeddingOptions.DownloadUri to a mirror of {ArchiveFileName}, or ModelDirectory to the onnx/ folder extracted " +
					$"from it, or put {ArchiveFileName} (SHA-256 {ArchiveSha256}) in {directory}.", e);
			}
			if (actualHash != ArchiveSha256)
				throw new InvalidDataException($"The file downloaded from {uri} has the SHA-256 hash {actualHash}, not {ArchiveSha256}: it was not kept.");
			Compat.MoveOverwrite(temporary, path);
			if (logger is not null)
				Log.Downloaded(logger, path, length, stopwatch.ElapsedMilliseconds);
		}
		finally
		{
			TryDelete(temporary);
		}
	}

	// Only the six files of onnx/, by their exact names: no other entry of the archive is written, so no path can leave the folder. The
	// files are renamed once all six are extracted.
	internal static async Task ExtractAsync(string archive, string onnx, ILogger? logger, CancellationToken cancellationToken)
	{
		var stopwatch = Stopwatch.StartNew();
		Directory.CreateDirectory(onnx);
		var suffix = "." + Guid.NewGuid().ToString("N") + ".tmp";
		var extracted = new HashSet<string>(StringComparer.Ordinal);
		try
		{
			// The file of onnx/ an entry is, or null for an entry that is not one of the six; a second one is an error.
			string? Accept(string entryName)
			{
				var name = entryName.Replace('\\', '/');
				if (name.StartsWith("./", StringComparison.Ordinal))
					name = name.Substring(2);
				var prefix = ExtractedFolderName + "/";
				if (!name.StartsWith(prefix, StringComparison.Ordinal))
					return null;
				var fileName = name.Substring(prefix.Length);
				if (Array.IndexOf(ModelFiles, fileName) < 0)
					return null;
				if (!extracted.Add(fileName))
					throw new InvalidDataException($"{archive} has {name} twice.");
				return Path.Combine(onnx, fileName + suffix);
			}

			using (var file = new FileStream(archive, FileMode.Open, FileAccess.Read, FileShare.Read, 81920, useAsync: true))
			using (var gzip = new GZipStream(file, CompressionMode.Decompress))
			{
#if NET
				using var tar = new TarReader(gzip);
				while (await tar.GetNextEntryAsync(copyData: false, cancellationToken).ConfigureAwait(false) is { } entry)
				{
					if (entry.EntryType is not (TarEntryType.RegularFile or TarEntryType.V7RegularFile) || entry.DataStream is null || Accept(entry.Name) is not { } target)
						continue;
					using var output = new FileStream(target, FileMode.CreateNew, FileAccess.Write, FileShare.None, 81920, useAsync: true);
					await entry.DataStream.CopyToAsync(output, cancellationToken).ConfigureAwait(false);
				}
#else
				using var tar = new ICSharpCode.SharpZipLib.Tar.TarInputStream(gzip, System.Text.Encoding.UTF8) { IsStreamOwner = false };
				while (tar.GetNextEntry() is { } entry)
				{
					cancellationToken.ThrowIfCancellationRequested();
					var type = entry.TarHeader.TypeFlag;
					if (entry.IsDirectory || (type != ICSharpCode.SharpZipLib.Tar.TarHeader.LF_NORMAL && type != ICSharpCode.SharpZipLib.Tar.TarHeader.LF_OLDNORM) || Accept(entry.Name) is not { } target)
						continue;
					using var output = new FileStream(target, FileMode.CreateNew, FileAccess.Write, FileShare.None, 81920, useAsync: true);
					await tar.CopyToAsync(output, 81920, cancellationToken).ConfigureAwait(false);
				}
#endif
			}
			var missing = ModelFiles.Where(name => !extracted.Contains(name)).ToArray();
			if (missing.Length > 0)
				throw new InvalidDataException($"{archive} has no {string.Join(", ", missing.Select(name => ExtractedFolderName + "/" + name))}.");
			foreach (var name in ModelFiles)
				Compat.MoveOverwrite(Path.Combine(onnx, name + suffix), Path.Combine(onnx, name));
		}
		finally
		{
			foreach (var name in ModelFiles)
				TryDelete(Path.Combine(onnx, name + suffix));
		}
		if (logger is not null)
			Log.Extracted(logger, onnx, stopwatch.ElapsedMilliseconds);
	}

	internal static async Task<string> HashFileAsync(string path, CancellationToken cancellationToken)
	{
		using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 81920, useAsync: true);
		using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
		var buffer = new byte[81920];
		int read;
		while ((read = await Compat.ReadAsync(file, buffer, cancellationToken).ConfigureAwait(false)) > 0)
			sha.AppendData(buffer, 0, read);
		return Compat.ToHex(sha.GetHashAndReset());
	}

	// The lock file stays: deleted, a process could lock the old file while another creates a new one. The open is tried again only
	// while another process (or another call of this one) holds the file; any other error ends at once.
	internal static async Task<FileStream> LockAsync(string path, CancellationToken cancellationToken)
	{
		while (true)
		{
			try
			{
				return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
			}
			catch (IOException e) when (IsHeldElsewhere(e))
			{
				await Task.Delay(100, cancellationToken).ConfigureAwait(false);
			}
			catch (Exception e) when (e is IOException or UnauthorizedAccessException)
			{
				throw CacheError(Path.GetDirectoryName(path)!, e);
			}
		}
	}

	// Windows: ERROR_SHARING_VIOLATION or ERROR_LOCK_VIOLATION. Linux and macOS: flock fails with EWOULDBLOCK, which .NET puts in HResult
	// as it is, 11 on Linux and 35 on macOS.
	private static bool IsHeldElsewhere(IOException e)
	{
		if (e.GetType() != typeof(IOException))
			return false;
		if (Compat.IsWindows)
			return (e.HResult & 0xFFFF) is 32 or 33;
		return Compat.IsMacOS ? e.HResult == 35 : e.HResult == 11;
	}

	private static void TryDelete(string path)
	{
		try
		{
			File.Delete(path);
		}
		catch (Exception e) when (e is IOException or UnauthorizedAccessException)
		{
			// A temporary file that cannot be deleted now has a unique name: it does not hide a file of the model.
		}
	}

	private static IOException CacheError(string directory, Exception e) =>
		new($"Cannot use {directory} for the model: {e.Message} Set ChromaDefaultEmbeddingOptions.CacheDirectory to another folder, or ModelDirectory to the extracted files.", e);
}
