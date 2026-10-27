using Microsoft.Extensions.Logging;

namespace ChromaDotNet.Embeddings;

// A file of a model repository of Hugging Face at a fixed revision: its path in the repository, and the SHA-256 hash of its content.
internal sealed class HuggingFaceFile
{
	public HuggingFaceFile(string path, string sha256, long size)
	{
		Path = path;
		Sha256 = sha256;
		Size = size;
	}

	public string Path { get; }

	public string Sha256 { get; }

	public long Size { get; }
}

// The files of a model, from a repository of Hugging Face at a fixed revision. In the cache they keep the layout of the repository,
// in the folder of the model under ~/.cache/chroma/onnx_models, and in a subfolder of it for one variant of the model.
internal sealed class HuggingFaceModel
{
	public HuggingFaceModel(string repository, string revision, string folder, string? subfolder, string optionsName, params HuggingFaceFile[] files)
	{
		Repository = repository;
		Revision = revision;
		Folder = folder;
		Subfolder = subfolder;
		OptionsName = optionsName;
		Files = files;
	}

	public string Repository { get; }

	public string Revision { get; }

	// The folder of the model under ~/.cache/chroma/onnx_models, like Qwen3-Embedding-0.6B.
	public string Folder { get; }

	// The folder of the variant in the folder of the model, like fp16, or null.
	public string? Subfolder { get; }

	// The options of the generator, for the messages.
	public string OptionsName { get; }

	public IReadOnlyList<HuggingFaceFile> Files { get; }

	// https://huggingface.co/{repository}/resolve/{revision}/{path}, or the same path on a mirror.
	public Uri FileUri(Uri? baseUri, HuggingFaceFile file)
	{
		var root = baseUri ?? new Uri(HuggingFaceCache.DefaultBaseUri);
		if (root.AbsoluteUri[root.AbsoluteUri.Length - 1] != '/')
			root = new Uri(root.AbsoluteUri + "/");
		return new Uri(root, $"{Repository}/resolve/{Revision}/{file.Path}");
	}
}

// The download of the files of a model of Hugging Face, as ModelCache downloads the default model: the files are used when they are all
// there with their sizes, without hashing them at each start; otherwise each file missing or of another size, like a copy cut short, is
// downloaded into a temporary file of its folder, checked against its SHA-256 hash, and renamed, so a file it downloads is whole. A lock
// file of the folder keeps a second process from downloading at the same time: it waits, then finds the files there.
internal static class HuggingFaceCache
{
	public const string DefaultBaseUri = "https://huggingface.co/";
	private const string LockFileName = "download.lock";

	// The folder of the files: ModelDirectory as it is, with the layout of the repository, or the folder of the cache, filled when files
	// are missing.
	public static async Task<string> EnsureAsync(HuggingFaceModel model, string? cacheDirectory, string? modelDirectory, Uri? baseUri, ILogger? logger, CancellationToken cancellationToken)
	{
		if (modelDirectory is not null)
		{
			var full = Path.GetFullPath(modelDirectory);
			var missing = model.Files.Where(file => !IsWhole(LocalPath(full, file), file)).Select(file => $"{file.Path} of {file.Size} bytes").ToArray();
			if (missing.Length > 0)
				throw new InvalidOperationException(
					$"{model.OptionsName}.ModelDirectory is {full}, which has no {string.Join(" and ", missing)}: set it to a folder with the files of " +
					$"{model.Repository} at the revision {model.Revision}, in the layout of the repository, or leave it null to download them.");
			return full;
		}

		var root = Path.GetFullPath(cacheDirectory ?? Path.Combine(ModelCache.ModelsRoot(model.OptionsName), model.Folder));
		var directory = model.Subfolder is null ? root : Path.Combine(root, model.Subfolder);
		if (HasAllFiles(model, directory))
			return directory;
		try
		{
			Directory.CreateDirectory(directory);
		}
		catch (Exception e) when (e is IOException or UnauthorizedAccessException)
		{
			throw ModelCache.CacheError(directory, e, model.OptionsName);
		}

		using (await ModelCache.LockAsync(Path.Combine(directory, LockFileName), cancellationToken, model.OptionsName).ConfigureAwait(false))
		{
			foreach (var file in model.Files)
			{
				// Another process may have downloaded it while this one waited for the lock.
				var path = LocalPath(directory, file);
				if (IsWhole(path, file))
					continue;
				Directory.CreateDirectory(Path.GetDirectoryName(path)!);
				var uri = model.FileUri(baseUri, file);
				await ModelCache.DownloadFileAsync(uri, path, file.Sha256, e => new InvalidOperationException(
					$"Could not download {file.Path} of {model.Repository} from {uri}: {e.Message} Without the network: set " +
					$"{model.OptionsName}.DownloadBaseUri to a mirror of Hugging Face, or ModelDirectory to a folder with the files of {model.Repository} " +
					$"at the revision {model.Revision}, or put {file.Path} (SHA-256 {file.Sha256}) in {directory}.", e), logger, cancellationToken).ConfigureAwait(false);
			}
		}
		return directory;
	}

	internal static bool HasAllFiles(HuggingFaceModel model, string directory) => model.Files.All(file => IsWhole(LocalPath(directory, file), file));

	// The file is there with the size it has at the revision: the hash is checked at the download, not at each start.
	private static bool IsWhole(string path, HuggingFaceFile file)
	{
		var info = new FileInfo(path);
		return info.Exists && info.Length == file.Size;
	}

	internal static string LocalPath(string directory, HuggingFaceFile file) => LocalPath(directory, file.Path);

	// A path of the repository, like onnx/model.onnx, in a folder with the layout of the repository.
	internal static string LocalPath(string directory, string path) => Path.Combine(directory, path.Replace('/', Path.DirectorySeparatorChar));
}
