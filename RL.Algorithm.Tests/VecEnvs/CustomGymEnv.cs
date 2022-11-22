namespace RL.Algorithm.Tests.VecEnvs;

public class CustomGymEnv : DigitalEnv
{
    private readonly int epLength;
    private int currentStep;
    private ndarray state = null!;
    private Random rnd = null!;

    public CustomGymEnv(DigitalSpace space)
        => (ActionSpace, ObservationSpace, currentStep, epLength, rnd) = (space, space, 0, 4, new());

    public override ndarray? Render(RanderMode randerMode)
    {
        if (randerMode == RanderMode.RGBArray)
            return np.zeros((4, 4, 3));
        return null;
    }

    public override ResetResult Reset(uint? seed = 0, Dictionary<string, object>? options = null)
    {
        rnd = new((int)(seed ??= 0));
        ObservationSpace.Seed(seed);
        currentStep = 0;
        ChooseNextState();
        return new(state, new());
    }

    public override StepResult Step(ndarray action)
    {
        ChooseNextState();
        currentStep++;
        return new(state, rnd.NextDouble(), currentStep >= epLength, false, new());
    }

    private static ndarray CustomMethod(int dim0 = 1, int dim1 = 1)
        => np.ones((dim0, dim1));

    private void ChooseNextState()
        => state = ObservationSpace.Sample();
}
