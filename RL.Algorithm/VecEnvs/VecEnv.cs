using RL.Env.Utils;

namespace RL.Algorithm.VecEnvs;

public abstract class VecEnv
{
    public DigitalSpace ActionSpace { get; init; }
    public DigitalSpace ObservationSpace { get; init; }
    public RewardRange RewardRange { get; init; } = new RewardRange(double.PositiveInfinity, double.NegativeInfinity);
    public int NumEnvs { get; init; }

    public VecEnv(BaseEnv<DigitalSpace>[] envs)
    {
        ArgumentNullException.ThrowIfNull(envs);
        if (!envs.Any())
            throw new ArgumentException("numEnvs cannot be empty.");
        NumEnvs = envs.Length;
        ActionSpace = envs[0].ActionSpace;
        ObservationSpace = envs[0].ObservationSpace;
        RewardRange = envs[0].RewardRange;
        Log.Information($"Number of Env: {envs.Length}, ActionSpace: {ActionSpace.GetType().Name}, ObservationSpace: {ObservationSpace.GetType().Name}, RewardRange: {RewardRange.Max}~{RewardRange.Min}", this);
    }

    public VecEnv(DigitalSpace actionSpace, DigitalSpace observationSpace, RewardRange rewardRange, int numEnvs)
    {
        ArgumentNullException.ThrowIfNull(actionSpace);
        ArgumentNullException.ThrowIfNull(observationSpace);
        ArgumentNullException.ThrowIfNull(rewardRange);
        ArgumentNullException.ThrowIfNull(numEnvs);
        ActionSpace = actionSpace;
        ObservationSpace = observationSpace;
        RewardRange = rewardRange;
        NumEnvs = numEnvs;
    }

    public abstract ResetResult Reset(uint? seed = null, Dictionary<string, object>? options = null);

    public abstract StepResult Step(ndarray action);

    public abstract ndarray? Render(RanderMode randerMode);

    public abstract void Close();
}
