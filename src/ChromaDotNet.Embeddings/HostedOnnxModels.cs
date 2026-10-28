using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ChromaDotNet.Embeddings;

// The ONNX export of Qwen3-Embedding-0.6B of onnx-community, one text at a time, as a decoder without a cache: input_ids, the
// attention mask of ones, the positions 0 to n-1, and the 28 pairs of past_key_values empty, [1, 8, 0, 128] in the type of the export
// (float16 or float32). Only last_hidden_state is asked; the embedding is its last token, the <|endoftext|> the tokenizer adds,
// normalized to length 1.
internal sealed class QwenOnnxModel : IDisposable
{
	private const string HiddenState = "last_hidden_state";

	private readonly InferenceSession _session;
	private readonly string[] _pastNames;
	private readonly OrtValue[] _pastValues;
	private readonly bool _halfOutput;

	private QwenOnnxModel(InferenceSession session, string[] pastNames, OrtValue[] pastValues, bool halfOutput, int dimensions)
	{
		_session = session;
		_pastNames = pastNames;
		_pastValues = pastValues;
		_halfOutput = halfOutput;
		Dimensions = dimensions;
	}

	public int Dimensions { get; }

	public static QwenOnnxModel Load(string modelPath, int intraOpNumThreads)
	{
		var session = HostedSession.Create(modelPath, intraOpNumThreads);
		var pastValues = new List<OrtValue>();
		try
		{
			foreach (var name in new[] { "input_ids", "attention_mask", "position_ids" })
			{
				if (!session.InputMetadata.ContainsKey(name))
					throw new InvalidDataException($"{modelPath} has no input {name}.");
			}
			if (!session.OutputMetadata.TryGetValue(HiddenState, out var hidden))
				throw new InvalidDataException($"{modelPath} has no output {HiddenState}.");
			var pastNames = session.InputMetadata.Keys.Where(name => name.StartsWith("past_key_values.", StringComparison.Ordinal)).ToArray();
			foreach (var name in pastNames)
			{
				// [batch, heads of the keys and values, past length, head size]: the batch of one text and no past.
				var metadata = session.InputMetadata[name];
				var shape = new long[] { 1, metadata.Dimensions[1], 0, metadata.Dimensions[3] };
				pastValues.Add(OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, metadata.ElementDataType, shape));
			}
			var halfOutput = hidden.ElementDataType == TensorElementType.Float16;
			if (!halfOutput && hidden.ElementDataType != TensorElementType.Float)
				throw new InvalidDataException($"{HiddenState} of {modelPath} is {hidden.ElementDataType}, not float or float16.");
			return new QwenOnnxModel(session, pastNames, pastValues.ToArray(), halfOutput, hidden.Dimensions[hidden.Dimensions.Length - 1]);
		}
		catch
		{
			foreach (var value in pastValues)
				value.Dispose();
			session.Dispose();
			throw;
		}
	}

	// The embedding of the ids of one text.
	public float[] Embed(long[] ids, CancellationToken cancellationToken)
	{
		var count = ids.Length;
		var mask = new long[count];
		var positions = new long[count];
		for (var i = 0; i < count; i++)
		{
			mask[i] = 1;
			positions[i] = i;
		}
		var shape = new long[] { 1, count };
		using var idsValue = OrtValue.CreateTensorValueFromMemory(ids, shape);
		using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, shape);
		using var positionsValue = OrtValue.CreateTensorValueFromMemory(positions, shape);
		var names = new List<string>(3 + _pastNames.Length) { "input_ids", "attention_mask", "position_ids" };
		names.AddRange(_pastNames);
		var values = new List<OrtValue>(names.Count) { idsValue, maskValue, positionsValue };
		values.AddRange(_pastValues);

		using var outputs = HostedSession.Run(_session, names, values, HiddenState, cancellationToken);
		var dimensions = Dimensions;
		var last = new float[dimensions];
		var offset = (count - 1) * dimensions;
		if (_halfOutput)
		{
			var hidden = outputs[0].GetTensorDataAsSpan<Float16>();
			for (var d = 0; d < dimensions; d++)
				last[d] = (float)hidden[offset + d];
		}
		else
		{
			outputs[0].GetTensorDataAsSpan<float>().Slice(offset, dimensions).CopyTo(last);
		}
		HostedSession.Normalize(last);
		return last;
	}

	public void Dispose()
	{
		foreach (var value in _pastValues)
			value.Dispose();
		_session.Dispose();
	}
}

// The ONNX export of SPLADE in the repository prithivida/Splade_PP_en_v1, one text at a time. Its inputs have the names of BERT:
// input_ids, input_mask for the attention mask, segment_ids for the token types (zeros for one text); its output is the logits of the
// masked language model, [1, tokens, 30522]. The value of a token of the vocabulary is the maximum over the tokens of the text of
// log(1 + ReLU(logit)), the pooling of SPLADE; the vector keeps the values above zero, in the order of the vocabulary.
internal sealed class SpladeOnnxModel : IDisposable
{
	private readonly InferenceSession _session;
	private readonly string _maskName;
	private readonly string? _typesName;
	private readonly string _outputName;

	private SpladeOnnxModel(InferenceSession session, string maskName, string? typesName, string outputName, int vocabularySize)
	{
		_session = session;
		_maskName = maskName;
		_typesName = typesName;
		_outputName = outputName;
		VocabularySize = vocabularySize;
	}

	public int VocabularySize { get; }

	public static SpladeOnnxModel Load(string modelPath, int intraOpNumThreads)
	{
		var session = HostedSession.Create(modelPath, intraOpNumThreads);
		try
		{
			var inputs = session.InputMetadata;
			if (!inputs.ContainsKey("input_ids"))
				throw new InvalidDataException($"{modelPath} has no input input_ids.");
			var mask = inputs.ContainsKey("input_mask") ? "input_mask" : inputs.ContainsKey("attention_mask") ? "attention_mask"
				: throw new InvalidDataException($"{modelPath} has no input input_mask or attention_mask.");
			var types = inputs.ContainsKey("segment_ids") ? "segment_ids" : inputs.ContainsKey("token_type_ids") ? "token_type_ids" : null;
			var output = session.OutputMetadata.First();
			if (output.Value.ElementDataType != TensorElementType.Float)
				throw new InvalidDataException($"{output.Key} of {modelPath} is {output.Value.ElementDataType}, not float.");
			return new SpladeOnnxModel(session, mask, types, output.Key, output.Value.Dimensions[output.Value.Dimensions.Length - 1]);
		}
		catch
		{
			session.Dispose();
			throw;
		}
	}

	// The indices, in ascending order, and the values of the sparse vector of the ids of one text.
	public (int[] Indices, float[] Values) Embed(long[] ids, CancellationToken cancellationToken)
	{
		var count = ids.Length;
		var mask = new long[count];
		for (var i = 0; i < count; i++)
			mask[i] = 1;
		var shape = new long[] { 1, count };
		using var idsValue = OrtValue.CreateTensorValueFromMemory(ids, shape);
		using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, shape);
		using var typesValue = OrtValue.CreateTensorValueFromMemory(new long[count], shape);
		var names = new List<string> { "input_ids", _maskName };
		var values = new List<OrtValue> { idsValue, maskValue };
		if (_typesName is not null)
		{
			names.Add(_typesName);
			values.Add(typesValue);
		}

		using var outputs = HostedSession.Run(_session, names, values, _outputName, cancellationToken);
		var logits = outputs[0].GetTensorDataAsSpan<float>();
		var size = VocabularySize;
		// log(1 + ReLU(x)) grows with x: its maximum over the tokens is the one of the largest logit above zero.
		var largest = new float[size];
		for (var t = 0; t < count; t++)
		{
			var row = logits.Slice(t * size, size);
			for (var j = 0; j < size; j++)
			{
				if (row[j] > largest[j])
					largest[j] = row[j];
			}
		}
		var indices = new List<int>();
		var vector = new List<float>();
		for (var j = 0; j < size; j++)
		{
			if (largest[j] <= 0)
				continue;
			var value = (float)Log1p(largest[j]);
			if (value > 0)
			{
				indices.Add(j);
				vector.Add(value);
			}
		}
		return (indices.ToArray(), vector.ToArray());
	}

	public void Dispose() => _session.Dispose();

	// log(1 + x) without the loss of 1 + x for a small x.
	internal static double Log1p(double x)
	{
		var u = 1.0 + x;
		return u == 1.0 ? x : Math.Log(u) * x / (u - 1.0);
	}
}

// The ONNX Runtime sessions of the hosted models: one text at a time, its operators on the threads given, and a run that a
// cancellation ends.
internal static class HostedSession
{
	public static InferenceSession Create(string modelPath, int intraOpNumThreads)
	{
		OnnxEnvironment.EnsureCreated();
		using var options = new SessionOptions
		{
			ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
			InterOpNumThreads = 1,
			IntraOpNumThreads = intraOpNumThreads,
		};
		return new InferenceSession(modelPath, options);
	}

	public static IDisposableReadOnlyCollection<OrtValue> Run(InferenceSession session, IReadOnlyCollection<string> names, IReadOnlyCollection<OrtValue> values, string output, CancellationToken cancellationToken)
	{
		cancellationToken.ThrowIfCancellationRequested();
		using var runOptions = new RunOptions();
		using var registration = cancellationToken.Register(() => runOptions.Terminate = true);
		try
		{
			return session.Run(runOptions, names, values, [output]);
		}
		catch (OnnxRuntimeException e) when (cancellationToken.IsCancellationRequested)
		{
			throw new OperationCanceledException("The computation of the embedding was cancelled.", e, cancellationToken);
		}
	}

	// The vector divided by its length, computed in double; a vector of zeros stays as it is.
	public static void Normalize(float[] vector)
	{
		double sum = 0;
		foreach (var x in vector)
			sum += (double)x * x;
		var norm = Math.Sqrt(sum);
		if (norm == 0)
			return;
		for (var i = 0; i < vector.Length; i++)
			vector[i] = (float)(vector[i] / norm);
	}
}
