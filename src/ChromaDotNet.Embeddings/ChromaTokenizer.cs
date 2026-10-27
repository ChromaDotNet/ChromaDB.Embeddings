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
		var tokenizer = TokenizerFile.Load(onnx, logger);
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
}

// The tokenizer of the hosted models of Chroma Cloud, one text at a time and without padding: the ids of the text, truncated to
// maxTokens as tokenizers truncates one sequence (the first tokens, then the last special token that the post-processor adds, [SEP] for
// SPLADE and <|endoftext|> for Qwen3-Embedding).
internal sealed class TruncatingTokenizer : IDisposable
{
	private readonly Tokenizer _tokenizer;
	private readonly uint _endId;

	private TruncatingTokenizer(Tokenizer tokenizer, uint endId, int maxTokens)
	{
		_tokenizer = tokenizer;
		_endId = endId;
		MaxTokens = maxTokens;
	}

	public int MaxTokens { get; }

	// The last special token of the post-processor, the one that ends a truncated text.
	public uint EndId => _endId;

	public static TruncatingTokenizer Load(string folder, int maxTokens, ILogger? logger)
	{
		var tokenizer = TokenizerFile.Load(folder, logger);
		try
		{
			var empty = tokenizer.Encode(string.Empty);
			if (empty.Length == 0)
				throw new InvalidDataException($"The tokenizer of {folder} adds no special token to the empty text.");
			return new TruncatingTokenizer(tokenizer, empty[empty.Length - 1], maxTokens);
		}
		catch
		{
			tokenizer.Dispose();
			throw;
		}
	}

	// The ids of a text, at most MaxTokens of them; tokens is the count before the truncation.
	public long[] Encode(string text, out int tokens)
	{
		var raw = _tokenizer.Encode(text);
		tokens = raw.Length;
		var count = Math.Min(raw.Length, MaxTokens);
		var ids = new long[count];
		for (var i = 0; i < count; i++)
			ids[i] = raw[i];
		if (raw.Length > MaxTokens)
			ids[count - 1] = _endId;
		return ids;
	}

	public void Dispose() => _tokenizer.Dispose();
}

// tokenizer.json as Tokenizers.DotNet loads it: without the truncation and the padding the file may have, which Tokenizers.DotNet has
// no call to change. A file that has neither is loaded as it is; otherwise a copy without them is written next to it.
internal static class TokenizerFile
{
	public const string CopyFileName = "tokenizer.untruncated.json";

	public static Tokenizer Load(string folder, ILogger? logger)
	{
		var original = Path.Combine(folder, "tokenizer.json");
		var copy = Untruncated(File.ReadAllBytes(original));
		if (copy is null)
			return new Tokenizer(vocabPath: original);

		var path = Path.Combine(folder, CopyFileName);
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
		try
		{
			return new Tokenizer(vocabPath: path);
		}
		finally
		{
			if (temporary is not null)
				File.Delete(temporary);
		}
	}

	// tokenizer.json with "truncation": null and "padding": null, the rest as it is; null when it has neither.
	internal static byte[]? Untruncated(byte[] tokenizerJson)
	{
		using (var document = JsonDocument.Parse(tokenizerJson))
		{
			if (document.RootElement.ValueKind != JsonValueKind.Object)
				throw new InvalidDataException("tokenizer.json is not a JSON object.");
			if (IsNullOrMissing(document.RootElement, "truncation") && IsNullOrMissing(document.RootElement, "padding"))
				return null;
		}
		var root = JsonNode.Parse(tokenizerJson)!.AsObject();
		root["truncation"] = null;
		root["padding"] = null;
		using var stream = new MemoryStream();
		using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }))
			root.WriteTo(writer);
		return stream.ToArray();
	}

	// The tokens of the vocabulary of a WordPiece or BPE tokenizer.json by id, from its model.vocab, as convert_ids_to_tokens of
	// transformers gives them: "##ing" for a continuation.
	internal static string[] Vocabulary(string folder, int size)
	{
		using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(folder, "tokenizer.json")));
		var tokens = new string[size];
		if (document.RootElement.TryGetProperty("model", out var model) && model.TryGetProperty("vocab", out var vocab) && vocab.ValueKind == JsonValueKind.Object)
		{
			foreach (var entry in vocab.EnumerateObject())
			{
				var id = entry.Value.GetInt32();
				if (id >= 0 && id < size)
					tokens[id] = entry.Name;
			}
		}
		for (var i = 0; i < size; i++)
		{
			if (tokens[i] is null)
				throw new InvalidDataException($"The vocabulary of {folder} has no token {i}.");
		}
		return tokens;
	}

	private static bool IsNullOrMissing(JsonElement root, string name) =>
		!root.TryGetProperty(name, out var value) || value.ValueKind == JsonValueKind.Null;

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
