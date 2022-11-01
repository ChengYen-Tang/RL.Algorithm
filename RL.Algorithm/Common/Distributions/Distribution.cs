namespace RL.Algorithm.Common.Distributions;

internal abstract class Distribution : IDistribution
{
    protected distributions.Distribution distribution = null!;

    public Tensor GetActions(bool deterministic = false)
        => deterministic ? Mode() : Sample();

    /// <summary>
    /// Continuous actions are usually considered to be independent,
    /// so we can sum components of the ``log_prob`` or the entropy.
    /// </summary>
    /// <param name="tensor"> shape: (n_batch, n_actions) or (n_batch,) </param>
    /// <returns> shape: (n_batch,) </returns>
    public static Tensor SumIndependentDims(Tensor tensor)
        => tensor.shape.Length > 1 ? tensor.sum(dim: 1) : tensor.sum();

    public abstract Tensor ActionsFromParams(IDictionary<string, object> kwargs);
    public abstract Tensor Entropy();
    public abstract Tensor LogProb(Tensor x);
    public abstract Tensor[] LogProbFromParams(IDictionary<string, object> kwargs);
    public abstract Tensor Mode();
    public abstract Tensor Sample();
}
