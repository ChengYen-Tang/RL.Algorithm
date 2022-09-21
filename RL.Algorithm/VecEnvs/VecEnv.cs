using BaseRLEnv;
using BaseRLEnv.Spaces;
using BaseRLEnv.Utils;
using NumpyDotNet;
using Serilog;

namespace RL.Algorithm.VecEnvs;

public abstract class VecEnv
{
    public DigitalSpace ActionSpace { get; protected init; }
    public DigitalSpace ObservationSpace { get; protected init; }
    public RewardRange RewardRange { get; protected init; } = new RewardRange(double.PositiveInfinity, double.NegativeInfinity);
    public BaseEnv<DigitalSpace>[] NumEnvs { get; init; } = null!;

    public VecEnv(BaseEnv<DigitalSpace>[] numEnvs)
    {
        ArgumentNullException.ThrowIfNull(numEnvs);
        if (!numEnvs.Any())
            throw new ArgumentException("numEnvs cannot be empty.");
        NumEnvs = numEnvs;
        ActionSpace = numEnvs[0].ActionSpace;
        ObservationSpace = numEnvs[0].ObservationSpace;
        RewardRange = numEnvs[0].RewardRange;
        Log.Information($"Number of Env: {numEnvs.Length}, ActionSpace: {ActionSpace.GetType().Name}, ObservationSpace: {ObservationSpace.GetType().Name}, RewardRange: {RewardRange.Max}~{RewardRange.Min}", this);
    }

    public abstract ResetResult Reset(uint? seed = null, Dictionary<string, dynamic>? options = null);

    public abstract StepResult Step(ndarray action);

    public abstract ndarray? Render(RanderMode randerMode);

    public abstract void Close();
}
