using Microsoft.ML.OnnxRuntime;

namespace ChromaDotNet.Embeddings;

// ONNX Runtime has one environment per process. The first session creates it at the warning level, and on Linux without a GPU it then
// warns at every start that the discovery of the GPU devices failed, though the model runs on the CPU. The package creates it first at
// the error level, unless the application created it before: then its level stays.
internal static class OnnxEnvironment
{
	private static readonly object Gate = new();

	internal static void EnsureCreated()
	{
		lock (Gate)
		{
			if (OrtEnv.IsCreated)
				return;
			var options = new EnvironmentCreationOptions { logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
			OrtEnv.CreateInstanceWithOptions(ref options);
		}
	}
}
