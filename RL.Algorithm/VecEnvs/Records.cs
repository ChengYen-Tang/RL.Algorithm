using NumpyDotNet;

namespace RL.Algorithm.VecEnvs;

public record ResetResult(ndarray Observation, Dictionary<string, dynamic>[] Info);
public record StepResult(ndarray Observation, double[] Reward, bool[] Terminated, bool[] Truncated, Dictionary<string, dynamic>[] Info);
