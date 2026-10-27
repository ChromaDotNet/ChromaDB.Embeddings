namespace ChromaDotNet.Embeddings;

// The files of the local models of the hosted embedding functions of Chroma Cloud, at fixed revisions of their repositories on Hugging
// Face. The hashes of the files of Git LFS are those of the repositories; the one of the tokenizer of SPLADE, a plain file of the
// repository, was computed on the file of that revision, whose Git blob is 327401998c676bd2d7c625039113b0577ea49fe7.
internal static class HostedModels
{
	public const string QwenRepository = "onnx-community/Qwen3-Embedding-0.6B-ONNX";
	public const string QwenRevision = "c25a394dd583836952667c12f008335071b3f43d";
	public const string QwenFolder = "Qwen3-Embedding-0.6B";
	public const string SpladeRepository = "prithivida/Splade_PP_en_v1";
	public const string SpladeRevision = "762be6a7206e2f299182705972a65e5c46e62be2";
	public const string SpladeFolder = "Splade_PP_en_v1";

	// The tokenizer of the ONNX export is the one of Qwen/Qwen3-Embedding-0.6B, byte for byte.
	private static readonly HuggingFaceFile QwenTokenizer = new("tokenizer.json", "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a", 11_423_705);

	public static HuggingFaceModel Qwen(ChromaQwenVariant variant) => variant switch
	{
		ChromaQwenVariant.Float16 => QwenFloat16,
		ChromaQwenVariant.Float32 => QwenFloat32,
		_ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "ChromaQwenEmbeddingOptions.Variant is Float16 or Float32."),
	};

	// The ONNX file of the model in a folder of its files.
	public static string QwenModelFile(ChromaQwenVariant variant) => variant == ChromaQwenVariant.Float16 ? "onnx/model_fp16.onnx" : "onnx/model.onnx";

	public static readonly HuggingFaceModel QwenFloat16 = new(QwenRepository, QwenRevision, QwenFolder, "fp16", nameof(ChromaQwenEmbeddingOptions),
		QwenTokenizer,
		new HuggingFaceFile("onnx/model_fp16.onnx", "f376cf065d912675dd559270e19361367ebc28c938b9811dd61d5834c78d0834", 583_522),
		new HuggingFaceFile("onnx/model_fp16.onnx_data", "06fbfded30166e3a33fe5252d7002ed1c370a4ec1bf45aba778cdf6cf7d31439", 1_199_927_296));

	public static readonly HuggingFaceModel QwenFloat32 = new(QwenRepository, QwenRevision, QwenFolder, "fp32", nameof(ChromaQwenEmbeddingOptions),
		QwenTokenizer,
		new HuggingFaceFile("onnx/model.onnx", "bf27b2f3f9ef9c32ca337d75b361fa99439deaeaefe82e4701b2dbd8439197cc", 307_161_415),
		new HuggingFaceFile("onnx/model.onnx_data", "f0a61604465929a27e68aa6217c8c89ec6186572f0209fdb7711adda48a9b9a9", 2_093_436_928));

	public static readonly HuggingFaceModel Splade = new(SpladeRepository, SpladeRevision, SpladeFolder, null, nameof(ChromaSpladeEmbeddingOptions),
		new HuggingFaceFile("tokenizer.json", SpladeTokenizerSha256, 711_649),
		new HuggingFaceFile("onnx/model.onnx", "0934583a27a031a66b2e847cbc260fbbef29689e969f500436460ef5146a43f2", 532_126_852));

	private const string SpladeTokenizerSha256 = "2fc687b11de0bc1b3d8348f92e3b49ef1089a621506c7661fbf3248fcd54947e";
}
