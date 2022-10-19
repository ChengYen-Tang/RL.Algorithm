namespace RL.Algorithm.VecEnvs;

public record ResetResult(ndarray Observation, Dictionary<string, object>[] Info);
public record StepResult(ndarray Observation, double[] Reward, bool[] Terminated, bool[] Truncated, Dictionary<string, object>[] Info);
