using Microsoft.Extensions.AI;

namespace ChromaDotNet.Embeddings;

/// <summary>
/// A sparse embedding: the values of some dimensions only, like the SPLADE embedding of a text, as the <c>SparseVector</c> of the Python
/// client of Chroma: indices in ascending order, a value for each index, and optionally a label for each index.
/// </summary>
public sealed class ChromaSparseEmbedding : Embedding
{
	private readonly int _dimensions;

	/// <summary>The embedding with the values of the dimensions in <paramref name="indices"/>.</summary>
	/// <param name="indices">The dimensions that have a value, at least 0, in strictly ascending order.</param>
	/// <param name="values">The value of each dimension of <paramref name="indices"/>.</param>
	/// <param name="labels">The label of each dimension, like its token, or null for none.</param>
	/// <param name="dimensions">The dimensions of the space, like the size of the vocabulary.</param>
	/// <exception cref="ArgumentException">The counts differ, or the indices are not in strictly ascending order.</exception>
	public ChromaSparseEmbedding(IReadOnlyList<int> indices, IReadOnlyList<float> values, IReadOnlyList<string>? labels = null, int dimensions = 0)
	{
		Compat.NotNull(indices, nameof(indices));
		Compat.NotNull(values, nameof(values));
		if (indices.Count != values.Count)
			throw new ArgumentException($"A sparse embedding needs as many indices as values: {indices.Count} indices and {values.Count} values.", nameof(values));
		if (labels is not null && labels.Count != indices.Count)
			throw new ArgumentException($"A sparse embedding needs as many labels as indices: {labels.Count} labels and {indices.Count} indices.", nameof(labels));
		for (var i = 0; i < indices.Count; i++)
		{
			if (indices[i] < 0 || (i > 0 && indices[i] <= indices[i - 1]))
				throw new ArgumentException($"The indices of a sparse embedding must be at least 0 and in strictly ascending order: {indices[i]} at position {i}.", nameof(indices));
		}
		if (dimensions < 0 || (dimensions > 0 && indices.Count > 0 && indices[indices.Count - 1] >= dimensions))
			throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "The dimensions of a sparse embedding must be 0, for unknown, or above its largest index.");
		Indices = Array.AsReadOnly(indices.ToArray());
		Values = Array.AsReadOnly(values.ToArray());
		Labels = labels is null ? null : Array.AsReadOnly(labels.ToArray());
		_dimensions = dimensions;
	}

	/// <summary>The dimensions that have a value, in ascending order.</summary>
	public IReadOnlyList<int> Indices { get; }

	/// <summary>The value of each dimension of <see cref="Indices"/>.</summary>
	public IReadOnlyList<float> Values { get; }

	/// <summary>The label of each dimension of <see cref="Indices"/>, like its token, or null.</summary>
	public IReadOnlyList<string>? Labels { get; }

	/// <summary>The dimensions of the space, like the size of the vocabulary, or 0 when unknown; <see cref="Indices"/> has those with a value.</summary>
	public override int Dimensions => _dimensions;
}
