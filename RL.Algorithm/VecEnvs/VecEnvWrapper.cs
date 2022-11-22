namespace RL.Algorithm.VecEnvs;

public abstract class VecEnvWrapper : VecEnv
{
    public VecEnv VEnv { get; init; }

    public VecEnvWrapper(VecEnv vEnv)
        : this(vEnv, vEnv.ActionSpace, vEnv.ObservationSpace) { }

    public VecEnvWrapper(VecEnv vEnv, DigitalSpace observationSpace)
        : this(vEnv, vEnv.ActionSpace, observationSpace) { }

    public VecEnvWrapper(VecEnv vEnv, DigitalSpace actionSpace, DigitalSpace observationSpace)
        : base(actionSpace, observationSpace, vEnv.RewardRange, vEnv.NumEnvs)
    {
        ArgumentNullException.ThrowIfNull(vEnv);
        VEnv = vEnv;
    }

    public override void Close()
    => VEnv.Close();

    public override ndarray? Render(RanderMode randerMode)
        => VEnv.Render(randerMode);
}
