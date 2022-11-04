﻿namespace RL.Algorithm.Common.Distributions;

/// <summary>
/// MultiCategorical distribution for multi discrete actions.
/// </summary>
internal class MultiCategoricalDistribution : Distribution, IProba
{
    private readonly long[] actionDims;
    private Categorical[] distribution = null!;

    public MultiCategoricalDistribution(long[] actionDims)
        => this.actionDims = actionDims;

    public override Tensor ActionsFromParams(IDictionary<string, object> kwargs)
    {
        ProbaDistribution(kwargs);
        bool deterministic = kwargs.ContainsKey("deterministic") ? (bool)kwargs["deterministic"] : false;
        return GetActions(deterministic);
    }

    public override Tensor? Entropy()
        => stack(distribution.Select(dist => dist.entropy()), dim: 1).sum(dim: 1);

    public override Tensor LogProb(Tensor actions)
        => stack(Enumerable.Zip(distribution, unbind(actions, dim: 1)).Select(item => item.First.log_prob(item.Second)), dim: 1).sum(dim: 1);

    public override Tensor[] LogProbFromParams(IDictionary<string, object> kwargs)
    {
        Tensor actions = ActionsFromParams(kwargs);
        Tensor logProb = LogProb(actions);
        return new[] { actions, logProb };
    }

    public override Tensor Mode()
        => stack(distribution.Select(dist => argmax(dist.probs, dim: 1)), dim: 1);

    public IProba ProbaDistribution(IDictionary<string, object> kwargs)
    {
        Tensor actionLogits = (kwargs["action_logits"] as Tensor)!;
        distribution = actionLogits.split(actionDims, dim: 1).Select(item => new Categorical(logits: item)).ToArray();
        return this;
    }

    public nn.Module ProbaDistributionNet(IDictionary<string, object> kwargs)
    {
        int latentDim = (int)kwargs["latent_dim"];
        return nn.Linear(latentDim, actionDims.Sum());
    }

    public override Tensor Sample()
        => stack(distribution.Select(dist => dist.sample()), dim: 1);
}
