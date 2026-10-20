using Microsoft.Extensions.Logging;

namespace ChromaDotNet.Embeddings;

internal static partial class Log
{
	[LoggerMessage(EventId = 1, Level = LogLevel.Information, Message = "Downloading the model from {Uri} into {Path}.")]
	public static partial void Downloading(ILogger logger, Uri uri, string path);

	[LoggerMessage(EventId = 2, Level = LogLevel.Information, Message = "Downloaded {Path}, {Length} bytes, in {Milliseconds} ms; its SHA-256 hash matches.")]
	public static partial void Downloaded(ILogger logger, string path, long length, long milliseconds);

	[LoggerMessage(EventId = 3, Level = LogLevel.Information, Message = "Using {Path}, already downloaded: its SHA-256 hash matches.")]
	public static partial void ArchiveFound(ILogger logger, string path);

	[LoggerMessage(EventId = 4, Level = LogLevel.Information, Message = "Extracted the model into {Directory} in {Milliseconds} ms.")]
	public static partial void Extracted(ILogger logger, string directory, long milliseconds);

	[LoggerMessage(EventId = 5, Level = LogLevel.Debug, Message = "Loaded the model of {Directory} with {Threads} threads in {Milliseconds} ms.")]
	public static partial void Loaded(ILogger logger, string directory, int threads, long milliseconds);

	[LoggerMessage(EventId = 6, Level = LogLevel.Warning, Message = "Cannot write {Path} ({Error}): the tokenizer is loaded from a temporary file.")]
	public static partial void TokenizerCopyNotWritten(ILogger logger, string path, string error);
}
