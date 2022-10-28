namespace RL.Algorithm.Tests.VecEnvs;

public class DummyRewardEnv : BaseEnv<DigitalSpace>
{
    private readonly int[] returnedRewards;
    private readonly int returnRewardIdx;
    private int t;

    public DummyRewardEnv(int returnRewardIdx = 0)
    {
        ActionSpace = new Discrete(2);
        ObservationSpace = new Box(-1, 1, new(1), np.Float32);
        returnedRewards = new[] { 0, 1, 3, 4 };
        this.returnRewardIdx = returnRewardIdx;
        t = returnRewardIdx;
    }

    public override ndarray? Render(RanderMode randerMode)
        => throw new NotImplementedException();

    public override ResetResult Reset(uint? seed = null, Dictionary<string, object>? options = null)
    {
        t = 0;
        return new(np.array(new float[] { returnedRewards[returnRewardIdx] }), new());
    }

    public override StepResult Step(ndarray action)
    {
        t++;
        int index = (t + returnRewardIdx) % returnedRewards.Length;
        int returnedValue = returnedRewards[index];
        return new(np.array(new float[] { returnedValue }), returnedValue, t == returnedRewards.Length, false, new());
    }
}
