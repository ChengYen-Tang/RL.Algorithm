namespace RL.Algorithm.Common.Distributions;

/// <summary>
/// Bernoulli distribution for MultiBinary action spaces.
/// </summary>
internal class BernoulliDistribution : Distribution, IProba
{
    private readonly int actionDims;
    private Bernoulli distribution = null!;

    public BernoulliDistribution(int actionDims)
        => this.actionDims = actionDims;

    public override Tensor ActionsFromParams(IDictionary<string, object> kwargs)
    {
        ProbaDistribution(kwargs);
        bool deterministic = kwargs.ContainsKey("deterministic") && (bool)kwargs["deterministic"];
        return GetActions(deterministic);
    }

    public override Tensor? Entropy()
        => distribution.entropy().sum(dim: 1);

    public override Tensor LogProb(Tensor actions)
        => distribution.log_prob(actions).sum(dim: 1);

    public override Tensor[] LogProbFromParams(IDictionary<string, object> kwargs)
    {
        Tensor actions = ActionsFromParams(kwargs);
        Tensor logProb = LogProb(actions);
        return new[] { actions, logProb };
    }

    public override Tensor Mode()
        => round(distribution.probs);

    public IProba ProbaDistribution(IDictionary<string, object> kwargs)
    {
        Tensor actionLogits = (kwargs["action_logits"] as Tensor)!;
        distribution = new(l: actionLogits);
        return this;
    }

    public nn.Module ProbaDistributionNet(IDictionary<string, object> kwargs)
    {
        int latentDim = (int)kwargs["latent_dim"];
        return nn.Linear(latentDim, actionDims);
    }

    public override Tensor Sample()
        => distribution.sample();
}
