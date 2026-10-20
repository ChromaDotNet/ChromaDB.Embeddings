using Microsoft.ML.OnnxRuntime;

namespace ChromaDotNet.Embeddings;

// ONNX Runtime has one environment per process. The first session creates it at the warning level, and on Linux without a GPU it then
// warns at every start that the discovery of the GPU devices failed, though the model runs on the CPU. The package creates it first at
// the error level, unless the application created it before: then its level stays.
internal static class OnnxEnvironment
{
	private static readonly object Gate = new();

	internal static void EnsureCreated() => EnsureCreated(() => OrtEnv.IsCreated, CreateAtErrorLevel);

	// The check and the creation as functions, so that a test can create the environment between them, as another part of the application
	// can: CreateInstanceWithOptions then throws, and the environment that exists stays, with its level. Any other failure goes on.
	internal static void EnsureCreated(Func<bool> isCreated, Action create)
	{
		lock (Gate)
		{
			if (isCreated())
				return;
			try
			{
				create();
			}
			catch (OnnxRuntimeException) when (isCreated())
			{
			}
		}
	}

	private static void CreateAtErrorLevel()
	{
		var options = new EnvironmentCreationOptions { logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
		OrtEnv.CreateInstanceWithOptions(ref options);
	}
}
