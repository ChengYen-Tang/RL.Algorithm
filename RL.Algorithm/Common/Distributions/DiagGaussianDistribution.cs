namespace RL.Algorithm.Common.Distributions;

/// <summary>
/// Gaussian distribution with diagonal covariance matrix, for continuous actions.
/// </summary>
internal class DiagGaussianDistribution : Distribution, IProbaWithParameter
{
    private readonly int actionDim;

    public DiagGaussianDistribution(int actionDim)
        => this.actionDim = actionDim;

    public (nn.Module, Parameter) ProbaDistributionNet(IDictionary<string, object> kwargs)
    {
        int latentDim = (int)kwargs["latent_dim"];
        double logStdInit = kwargs.ContainsKey("log_std_init") ? (double)kwargs["log_std_init"] : 0.0;
        return (nn.Linear(latentDim, actionDim), nn.Parameter(ones(actionDim) * logStdInit, true));
    }

    public IProbaWithParameter ProbaDistribution(IDictionary<string, object> kwargs)
    {
        Tensor meanActions = (kwargs["mean_actions"] as Tensor)!;
        Tensor logStd = (kwargs["log_std"] as Tensor)!;
        Tensor actionStd = ones_like(meanActions) * logStd.exp();
        distribution = new Normal(meanActions, actionStd);
        return this;
    }

    public override Tensor Sample()
        => distribution.rsample();

    public override Tensor ActionsFromParams(IDictionary<string, object> kwargs)
    {
        ProbaDistribution(kwargs);
        bool deterministic = kwargs.ContainsKey("deterministic") ? (bool)kwargs["deterministic"] : false;
        return GetActions(deterministic);
    }

    public override Tensor Entropy()
        => SumIndependentDims(distribution.entropy());

    public override Tensor LogProb(Tensor actions)
    {
        Tensor logProb = distribution.log_prob(actions);
        return SumIndependentDims(logProb);
    }

    public override Tensor[] LogProbFromParams(IDictionary<string, object> kwargs)
    {
        Tensor meanActions = (kwargs["mean_actions"] as Tensor)!;
        Tensor logStd = (kwargs["log_std"] as Tensor)!;
        Tensor actions = ActionsFromParams(new Dictionary<string, object>() { { "mean_actions", meanActions }, { "log_std", logStd } });
        Tensor logProb = LogProb(actions);
        return new[] { actions, logProb };
    }

    public override Tensor Mode()
        => distribution.mean;
}
