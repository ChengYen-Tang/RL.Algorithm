namespace RL.Algorithm.Common.Distributions;

/// <summary>
/// Gaussian distribution with diagonal covariance matrix, for continuous actions.
/// </summary>
internal class DiagGaussianDistribution : Distribution, IProbaWithParameter
{
    private readonly int actionDim;
    public Normal Distribution { get; private set; } = null!;

    /// <summary></summary>
    /// <param name="actionDim"> Dimension of the action space. </param>
    public DiagGaussianDistribution(int actionDim)
        => this.actionDim = actionDim;

    public (nn.Module, Parameter) ProbaDistributionNet(IDictionary<string, object> kwargs)
    {
        int latentDim = (int)kwargs["latent_dim"];
        Tensor logStdInit = kwargs.ContainsKey("log_std_init") ? (kwargs["log_std_init"] as Tensor)! : 0.0;
        return (nn.Linear(latentDim, actionDim), nn.Parameter(ones(actionDim) * logStdInit, true));
    }

    public virtual IProbaWithParameter ProbaDistribution(IDictionary<string, object> kwargs)
    {
        Tensor meanActions = (kwargs["mean_actions"] as Tensor)!;
        Tensor logStd = (kwargs["log_std"] as Tensor)!;
        Tensor actionStd = ones_like(meanActions) * logStd.exp();
        Distribution = new(meanActions, actionStd);
        return this;
    }

    public override Tensor Sample()
        => Distribution.rsample();

    public override Tensor ActionsFromParams(IDictionary<string, object> kwargs)
    {
        ProbaDistribution(kwargs);
        bool deterministic = kwargs.ContainsKey("deterministic") && (bool)kwargs["deterministic"];
        return GetActions(deterministic);
    }

    public override Tensor? Entropy()
        => SumIndependentDims(Distribution.entropy());

    public override Tensor LogProb(Tensor actions)
    {
        Tensor logProb = Distribution.log_prob(actions);
        return SumIndependentDims(logProb);
    }

    public override Tensor[] LogProbFromParams(IDictionary<string, object> kwargs)
    {
        Tensor actions = ActionsFromParams(kwargs);
        Tensor logProb = LogProb(actions);
        return new[] { actions, logProb };
    }

    public override Tensor Mode()
        => Distribution.mean;
}
