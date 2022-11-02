namespace RL.Algorithm.Common.Distributions;

/// <summary>
/// Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.
/// </summary>
internal class SquashedDiagGaussianDistribution : DiagGaussianDistribution
{
    private readonly double epsilon;
    private Tensor gaussianActions = null!;

    public SquashedDiagGaussianDistribution(int actionDim, double epsilon = 1e-6)
        : base(actionDim) => this.epsilon = epsilon;

    public override IProbaWithParameter ProbaDistribution(IDictionary<string, object> kwargs)
        => base.ProbaDistribution(kwargs);

    public override Tensor LogProb(Tensor actions)
        => LogProb(actions, null);

    public Tensor LogProb(Tensor actions, Tensor? gaussianActions = null)
    {
        gaussianActions ??= TanhBijector.Inverse(actions);
        Tensor logProb = base.LogProb(gaussianActions);
        logProb -= sum(log(1 - actions.pow(2) + epsilon), dim: 1);
        return logProb;
    }

    public override Tensor? Entropy()
        => null!;

    public override Tensor Sample()
    {
        gaussianActions = base.Mode();
        return tanh(gaussianActions);
    }

    public override Tensor Mode()
    {
        gaussianActions = base.Mode();
        return tanh(gaussianActions);
    }

    public override Tensor[] LogProbFromParams(IDictionary<string, object> kwargs)
    {
        Tensor action = ActionsFromParams(kwargs);
        Tensor logProb = LogProb(action, gaussianActions);
        return new Tensor[] { action, logProb };
    }
}
