namespace RL.Algorithm.Tests.VecEnvs;

public class NanAndInfEnv : DigitalEnv
{
    public NanAndInfEnv()
    {
        ActionSpace = new Box(-np.Inf, np.Inf, new(1), np.Float64);
        ObservationSpace = new Box(-np.Inf, np.Inf, new(1), np.Float64);
    }

    public override ndarray? Render(RanderMode randerMode)
        => throw new NotImplementedException();

    public override ResetResult Reset(uint? seed = null, Dictionary<string, object>? options = null)
        => new(np.array(new double[] { 0.0 }), new());

    public override StepResult Step(ndarray action)
    {
        double obs = np.allb(action > 0) ?
            np.NaN : np.allb(action < 0) ?
            np.Inf : 0;
        return new(np.array(new double[] { obs }), 0, false, false, new());
    }
}
