using NUnit.Framework;

namespace ChromaDotNet.Embeddings.Tests;

// The vectors of Python for the same texts: the 135 texts of corpus.json (empty, whitespace, accents, emoji, CJK, other scripts,
// symbols, control characters, texts of 254 to 1000 tokens, 100 short documents) and the texts of python-special.json.
[TestFixture]
public class ParityTests
{
	// The largest difference of a component measured on the 135 texts was 1.863e-07, about 1.6 times the float32 epsilon (1.19e-07),
	// from the order of the sums in ONNX Runtime 1.24.4 here and 1.31.0 in Python. 1e-6 leaves room for other CPUs and thread counts,
	// and is still far below a difference of meaning: the lowest cosine measured is 0.99999994.
	private const float Tolerance = 1e-6f;

	[Test]
	public async Task IdsAreThoseOfPython()
	{
		var generator = await SharedGenerator.GetAsync();
		var texts = Reference.Texts;
		var different = new List<string>();
		for (var i = 0; i < texts.Length; i++)
		{
			var (ids, _) = Encode(generator, texts[i]);
			if (!ids.SequenceEqual(Reference.PythonIds[i]))
				different.Add($"{i} ({Reference.Kinds[i]})");
		}
		Assert.That(different, Is.Empty, "Texts with other ids than Python");
		TestContext.Out.WriteLine($"ids identical: {texts.Length}/{texts.Length}");
	}

	[Test]
	public async Task VectorsAreThoseOfPython()
	{
		var generator = await SharedGenerator.GetAsync();
		var embeddings = await generator.GenerateAsync(Reference.Texts);
		var vectors = embeddings.Select(e => e.Vector.ToArray()).ToArray();
		var python = Reference.PythonVectors;
		Assert.That(vectors, Has.Length.EqualTo(python.Length));

		var maxDiff = 0f;
		var minCosine = 1d;
		for (var i = 0; i < vectors.Length; i++)
		{
			Assert.That(vectors[i], Has.Length.EqualTo(ChromaDefaultEmbeddingGenerator.Dimensions));
			for (var j = 0; j < vectors[i].Length; j++)
				maxDiff = Math.Max(maxDiff, Math.Abs(vectors[i][j] - python[i][j]));
			minCosine = Math.Min(minCosine, Cosine(vectors[i], python[i]));
		}
		var sameNeighbours = Enumerable.Range(0, vectors.Length).Count(i => Neighbours(vectors, i).SequenceEqual(Neighbours(python, i)));
		TestContext.Out.WriteLine($"max abs diff {maxDiff:E3}, lowest cosine {minCosine:G10} (1 - cosine = {1 - minCosine:E2}), same top-5 neighbours {sameNeighbours}/{vectors.Length}");
		Assert.Multiple(() =>
		{
			Assert.That(maxDiff, Is.LessThanOrEqualTo(Tolerance));
			Assert.That(minCosine, Is.GreaterThan(0.9999990));
			Assert.That(sameNeighbours, Is.EqualTo(vectors.Length));
		});
	}

	// [CLS], [SEP], [PAD], [MASK] and [UNK] in the text are special tokens, as in Python; the lowercase ones are not. NUL and other
	// control characters are dropped by the normalizer, as in Python, also inside a text.
	[Test]
	public async Task SpecialTokensAndControlCharactersAreThoseOfPython()
	{
		var generator = await SharedGenerator.GetAsync();
		var (texts, pythonIds, pythonVectors) = Reference.Special;
		var embeddings = await generator.GenerateAsync(texts);
		var maxDiff = 0f;
		Assert.Multiple(() =>
		{
			for (var i = 0; i < texts.Length; i++)
			{
				var (ids, _) = Encode(generator, texts[i]);
				Assert.That(ids, Is.EqualTo(pythonIds[i]), $"ids of {Escape(texts[i])}");
				var vector = embeddings[i].Vector.Span;
				for (var j = 0; j < vector.Length; j++)
					maxDiff = Math.Max(maxDiff, Math.Abs(vector[j] - pythonVectors[i][j]));
			}
		});
		TestContext.Out.WriteLine($"{texts.Length} texts: ids identical, max abs diff {maxDiff:E3}");
		Assert.That(maxDiff, Is.LessThanOrEqualTo(Tolerance));
	}

	// A batch of 32 gives the same vectors as each text alone, to the last bit: every text is padded to 256 tokens, as in Python.
	[Test]
	public async Task BatchesGiveTheVectorsOfEachTextAlone()
	{
		var generator = await SharedGenerator.GetAsync();
		var texts = Reference.Texts;
		var together = await generator.GenerateAsync(texts);
		var different = 0;
		var maxDiff = 0f;
		for (var i = 0; i < texts.Length; i++)
		{
			var alone = (await generator.GenerateAsync([texts[i]]))[0].Vector.ToArray();
			var batched = together[i].Vector.ToArray();
			if (!alone.SequenceEqual(batched))
				different++;
			for (var j = 0; j < alone.Length; j++)
				maxDiff = Math.Max(maxDiff, Math.Abs(alone[j] - batched[j]));
		}
		TestContext.Out.WriteLine($"batch vs alone: {texts.Length - different}/{texts.Length} identical, max abs diff {maxDiff:E3}");
		Assert.That(different, Is.Zero);
	}

	// Texts of 255, 256, 257 and 1000 tokens with [CLS] and [SEP]: at most 256, the last [SEP], as tokenizers truncates in Python.
	[TestCase(254)]
	[TestCase(255)]
	[TestCase(256)]
	[TestCase(257)]
	[TestCase(300)]
	[TestCase(1000)]
	public async Task LongTextsAreTruncatedAsInPython(int tokens)
	{
		var generator = await SharedGenerator.GetAsync();
		var index = Array.IndexOf(Reference.Kinds, $"long{tokens}");
		Assert.That(index, Is.GreaterThanOrEqualTo(0));
		var text = Reference.Texts[index];
		Assert.That(generator.Tokenizer.CountTokens(text), Is.EqualTo(tokens), "the tokens of the text without truncation");

		var (ids, mask) = Encode(generator, text);
		var real = Math.Min(tokens, ChromaDefaultEmbeddingGenerator.MaxTokens);
		Assert.Multiple(() =>
		{
			Assert.That(ids, Is.EqualTo(Reference.PythonIds[index]));
			Assert.That(mask.Count(m => m == 1), Is.EqualTo(real));
			Assert.That(mask.Take(real), Is.All.EqualTo(1));
			Assert.That(ids[0], Is.EqualTo(101), "[CLS]");
			Assert.That(ids[real - 1], Is.EqualTo(102), "[SEP]");
			Assert.That(ids.Skip(real), Is.All.EqualTo(0), "[PAD]");
		});
	}

	// The empty text and a text of whitespace are [CLS] [SEP], as in Python.
	[TestCase("")]
	[TestCase("   \t \n ")]
	public async Task EmptyTextIsClsSep(string text)
	{
		var generator = await SharedGenerator.GetAsync();
		var (ids, mask) = Encode(generator, text);
		Assert.That(ids.Take(3), Is.EqualTo(new long[] { 101, 102, 0 }));
		Assert.That(mask.Sum(), Is.EqualTo(2));
	}

	private static (long[] Ids, long[] Mask) Encode(ChromaDefaultEmbeddingGenerator generator, string text)
	{
		var ids = new long[ChromaDefaultEmbeddingGenerator.MaxTokens];
		var mask = new long[ChromaDefaultEmbeddingGenerator.MaxTokens];
		generator.Tokenizer.Encode(text, ids, mask);
		return (ids, mask);
	}

	private static double Cosine(float[] a, float[] b)
	{
		double dot = 0, na = 0, nb = 0;
		for (var i = 0; i < a.Length; i++)
		{
			dot += (double)a[i] * b[i];
			na += (double)a[i] * a[i];
			nb += (double)b[i] * b[i];
		}
		return dot / Math.Sqrt(na * nb);
	}

	// The 5 nearest other texts by cosine.
	private static int[] Neighbours(float[][] vectors, int i) => Enumerable.Range(0, vectors.Length).Where(j => j != i)
		.OrderByDescending(j => Cosine(vectors[i], vectors[j])).Take(5).ToArray();

	private static string Escape(string text) => string.Concat(text.Select(c => char.IsControl(c) ? $"\\u{(int)c:x4}" : c.ToString()));
}
