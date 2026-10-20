using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Logging;
using Tokenizers.DotNet;

namespace ChromaDotNet.Embeddings;

// The tokenizer of Python: Tokenizer.from_file("tokenizer.json") of Hugging Face tokenizers, then enable_truncation(max_length=256) and
// enable_padding(pad_id=0, pad_token="[PAD]", length=256). tokenizer.json has a truncation and a padding of its own, to 128 tokens, and
// Tokenizers.DotNet loads only a file and has no call to change them: a copy without them is loaded, and the truncation and the padding
// to 256 are done here, as tokenizers does them for one sequence (longest_first, right, stride 0): the first 254 tokens of the text
// between [CLS] and [SEP].
internal sealed class ChromaTokenizer : IDisposable
{
	public const string CopyFileName = "tokenizer.untruncated.json";
	private const int PadId = 0;

	private readonly Tokenizer _tokenizer;
	private readonly long _sepId;

	private ChromaTokenizer(Tokenizer tokenizer, long sepId)
	{
		_tokenizer = tokenizer;
		_sepId = sepId;
	}

	public static ChromaTokenizer Load(string onnx, ILogger? logger)
	{
		var copy = Untruncated(File.ReadAllBytes(Path.Combine(onnx, "tokenizer.json")));
		var path = Path.Combine(onnx, CopyFileName);
		string? temporary = null;
		try
		{
			if (!File.Exists(path) || !File.ReadAllBytes(path).AsSpan().SequenceEqual(copy))
				WriteAndRename(path, copy);
		}
		catch (Exception e) when (e is IOException or UnauthorizedAccessException)
		{
			// A ModelDirectory that cannot be written, or a copy another process holds: a file of this process, deleted after the load.
			if (logger is not null)
				Log.TokenizerCopyNotWritten(logger, path, e.Message);
			temporary = Path.Combine(Path.GetTempPath(), "chromadotnet-" + Guid.NewGuid().ToString("N") + "-" + CopyFileName);
			File.WriteAllBytes(temporary, copy);
			path = temporary;
		}

		Tokenizer tokenizer;
		try
		{
			tokenizer = new Tokenizer(vocabPath: path);
		}
		finally
		{
			if (temporary is not null)
				File.Delete(temporary);
		}
		try
		{
			// The empty text is [CLS] [SEP] in Python: the post-processor of tokenizer.json adds them, and the copy pads nothing.
			var empty = tokenizer.Encode(string.Empty);
			if (empty.Length != 2)
				throw new InvalidDataException($"The tokenizer of {onnx} gives {empty.Length} tokens for the empty text, not the 2 of [CLS] and [SEP].");
			return new ChromaTokenizer(tokenizer, empty[1]);
		}
		catch
		{
			tokenizer.Dispose();
			throw;
		}
	}

	// tokenizer.json with "truncation": null and "padding": null, the rest as it is.
	internal static byte[] Untruncated(byte[] tokenizerJson)
	{
		if (JsonNode.Parse(tokenizerJson) is not JsonObject root)
			throw new InvalidDataException("tokenizer.json is not a JSON object.");
		root["truncation"] = null;
		root["padding"] = null;
		using var stream = new MemoryStream();
		using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }))
			root.WriteTo(writer);
		return stream.ToArray();
	}

	// The ids and the attention mask of a text into ids and mask, of MaxTokens each, all zeros; returns the count of real tokens.
	public int Encode(string text, long[] ids, long[] mask)
	{
		var tokens = _tokenizer.Encode(text);
		var count = Math.Min(tokens.Length, ChromaDefaultEmbeddingGenerator.MaxTokens);
		for (var i = 0; i < count; i++)
		{
			ids[i] = tokens[i];
			mask[i] = 1;
		}
		if (tokens.Length > ChromaDefaultEmbeddingGenerator.MaxTokens)
			ids[count - 1] = _sepId;
		for (var i = count; i < ids.Length; i++)
		{
			ids[i] = PadId;
			mask[i] = 0;
		}
		return count;
	}

	// The tokens of a text without the truncation, [CLS] and [SEP] included.
	public int CountTokens(string text) => _tokenizer.Encode(text).Length;

	public void Dispose() => _tokenizer.Dispose();

	private static void WriteAndRename(string path, byte[] content)
	{
		var temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
		try
		{
			File.WriteAllBytes(temporary, content);
			Compat.MoveOverwrite(temporary, path);
		}
		finally
		{
			if (File.Exists(temporary))
				File.Delete(temporary);
		}
	}
}
