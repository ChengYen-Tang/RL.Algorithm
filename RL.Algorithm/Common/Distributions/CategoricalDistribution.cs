namespace RL.Algorithm.Common.Distributions;

/// <summary>
/// Categorical distribution for discrete actions.
/// </summary>
internal class CategoricalDistribution : Distribution, IProba
{
    private readonly int actionDim;
    public Categorical Distribution { get; private set; } = null!;

    public CategoricalDistribution(int actionDim)
        => this.actionDim = actionDim;

    public nn.Module ProbaDistributionNet(IDictionary<string, object> kwargs)
    {
        int latentDim = (int)kwargs["latent_dim"];
        return nn.Linear(latentDim, actionDim);
    }

    public IProba ProbaDistribution(IDictionary<string, object> kwargs)
    {
        Tensor actionLogits = (kwargs["action_logits"] as Tensor)!;
        Distribution = new(logits: actionLogits);
        return this;
    }

    public override Tensor ActionsFromParams(IDictionary<string, object> kwargs)
    {
        ProbaDistribution(kwargs);
        bool deterministic = kwargs.ContainsKey("deterministic") && (bool)kwargs["deterministic"];
        return GetActions(deterministic);
    }

    public override Tensor? Entropy()
        => Distribution.entropy();

    public override Tensor LogProb(Tensor actions)
        => Distribution.log_prob(actions);

    public override Tensor[] LogProbFromParams(IDictionary<string, object> kwargs)
    {
        Tensor actions = ActionsFromParams(kwargs);
        Tensor logProb = LogProb(actions);
        return new[] { actions, logProb };
    }

    public override Tensor Mode()
        => argmax(Distribution.probs, dim: 1);

    public override Tensor Sample()
        => Distribution.sample();
}
