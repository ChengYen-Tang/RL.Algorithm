namespace RL.Algorithm.Tests.VecEnvs;

public class StepEnv : BaseEnv<DigitalSpace>
{
    private readonly int maxSteps;
    private int currentStep;

    public StepEnv(int maxSteps)
    {
        ActionSpace = new Discrete(2);
        ObservationSpace = new Box(0, 999, new(1), np.Int32);
        this.maxSteps = maxSteps;
        currentStep = 0;
    }

    public override ndarray? Render(RanderMode randerMode)
        => throw new NotImplementedException();

    public override ResetResult Reset(uint? seed = null, Dictionary<string, object>? options = null)
    {
        currentStep = 0;
        return new(np.array(new int[] { currentStep }), new());
    }

    public override StepResult Step(ndarray action)
    {
        int prevStep = currentStep;
        currentStep++;
        bool done = currentStep >= maxSteps;
        return new(np.array(new int[] { prevStep }), 0.0, done, false, new());
    }
}
