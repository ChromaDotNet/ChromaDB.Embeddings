#if !NET
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ChromaDotNet.Embeddings;

// .NET Framework only: on .NET 8 and later the model is OnnxEmbeddingModel of ElBruno.LocalEmbeddings, which this class follows, with
// the same calls. As in Python: input_ids, attention_mask and token_type_ids of zeros; the mean of last_hidden_state over the tokens of
// the mask; L2 normalization, with a norm of 0 taken as 1e-12.
internal sealed class OnnxEmbeddingModel : IDisposable
{
	private InferenceSession? _session;
	private string _output = "last_hidden_state";
	private bool _normalize;

	public int EmbeddingDimension { get; private set; }

	public void Load(string modelPath, bool normalizeEmbeddings, bool useParallelExecution, int? interOpNumThreads, int? intraOpNumThreads)
	{
		using var options = new SessionOptions { ExecutionMode = useParallelExecution ? ExecutionMode.ORT_PARALLEL : ExecutionMode.ORT_SEQUENTIAL };
		if (interOpNumThreads is { } inter)
			options.InterOpNumThreads = inter;
		if (intraOpNumThreads is { } intra)
			options.IntraOpNumThreads = intra;
		_session = new InferenceSession(modelPath, options);
		if (!_session.OutputMetadata.ContainsKey(_output))
			_output = _session.OutputMetadata.Keys.First();
		var dimensions = _session.OutputMetadata[_output].Dimensions;
		EmbeddingDimension = dimensions[dimensions.Length - 1];
		_normalize = normalizeEmbeddings;
	}

	public float[][] GenerateEmbeddings(long[][] inputIds, long[][] attentionMasks, CancellationToken cancellationToken)
	{
		var session = _session ?? throw new InvalidOperationException("The model is not loaded.");
		cancellationToken.ThrowIfCancellationRequested();
		var count = inputIds.Length;
		if (count == 0)
			return [];
		var length = inputIds[0].Length;
		var ids = new DenseTensor<long>([count, length]);
		var mask = new DenseTensor<long>([count, length]);
		var types = new DenseTensor<long>([count, length]);
		for (var i = 0; i < count; i++)
		{
			inputIds[i].AsSpan().CopyTo(ids.Buffer.Span.Slice(i * length, length));
			attentionMasks[i].AsSpan().CopyTo(mask.Buffer.Span.Slice(i * length, length));
		}
		var inputs = new List<NamedOnnxValue>
		{
			NamedOnnxValue.CreateFromTensor("input_ids", ids),
			NamedOnnxValue.CreateFromTensor("attention_mask", mask),
		};
		if (session.InputMetadata.ContainsKey("token_type_ids"))
			inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", types));

		using var results = session.Run(inputs, [_output]);
		var hidden = results[0].AsTensor<float>().ToDenseTensor().Buffer.Span;
		var dimension = EmbeddingDimension;
		var embeddings = new float[count][];
		for (var i = 0; i < count; i++)
		{
			var sum = new float[dimension];
			var tokens = 0f;
			for (var t = 0; t < length; t++)
			{
				if (attentionMasks[i][t] == 0)
					continue;
				tokens++;
				var row = hidden.Slice((i * length + t) * dimension, dimension);
				for (var d = 0; d < dimension; d++)
					sum[d] += row[d];
			}
			var divisor = Math.Max(tokens, 1e-9f);
			for (var d = 0; d < dimension; d++)
				sum[d] /= divisor;
			if (_normalize)
			{
				var squares = 0f;
				for (var d = 0; d < dimension; d++)
					squares += sum[d] * sum[d];
				var norm = (float)Math.Sqrt(squares);
				if (norm == 0)
					norm = 1e-12f;
				for (var d = 0; d < dimension; d++)
					sum[d] /= norm;
			}
			embeddings[i] = sum;
		}
		return embeddings;
	}

	public void Dispose() => _session?.Dispose();
}
#endif
