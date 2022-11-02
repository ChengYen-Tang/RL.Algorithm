namespace RL.Algorithm.Common.Distributions;

internal interface IDistribution
{
    /// <summary>
    /// Returns the log likelihood
    /// </summary>
    /// <param name="x"> the taken action </param>
    /// <returns> The log likelihood of the distribution </returns>
    Tensor LogProb(Tensor x);
    /// <summary>
    /// Returns Shannon's entropy of the probability
    /// </summary>
    /// <returns> the entropy, or None if no analytical form is known </returns>
    Tensor? Entropy();
    /// <summary>
    /// Returns a sample from the probability distribution
    /// </summary>
    /// <returns> the stochastic action </returns>
    Tensor Sample();
    /// <summary>
    /// Returns the most likely action(deterministic output)
    /// from the probability distribution
    /// </summary>
    /// <returns> the stochastic action </returns>
    Tensor Mode();
    /// <summary>
    /// Return actions according to the probability distribution.
    /// </summary>
    /// <param name="deterministic"></param>
    /// <returns></returns>
    Tensor GetActions(bool deterministic);
    /// <summary>
    /// Returns samples from the probability distribution
    ///    given its parameters.
    /// </summary>
    /// <param name="kwargs"></param>
    /// <returns> actions </returns>
    Tensor ActionsFromParams(IDictionary<string, object> kwargs);
    /// <summary>
    /// Returns samples and the associated log probabilities
    /// from the probability distribution given its parameters.
    /// </summary>
    /// <param name="kwargs"></param>
    /// <returns> actions and log prob </returns>
    Tensor[] LogProbFromParams(IDictionary<string, object> kwargs);
}

internal interface IProba : IDistribution
{
    /// <summary>
    /// Create the layers and parameters that represent the distribution.
    /// 
    /// Subclasses must define this, but the arguments and return type vary between concrete classes.
    /// </summary>
    /// <param name="kwargs"></param>
    /// <returns></returns>
    nn.Module ProbaDistributionNet(IDictionary<string, object> kwargs);
    /// <summary>
    /// Set parameters of the distribution.
    /// </summary>
    /// <param name="kwargs"></param>
    /// <returns> self </returns>
    IProba ProbaDistribution(IDictionary<string, object> kwargs);
}

internal interface IProbaWithParameter : IDistribution
{
    /// <summary>
    /// Create the layers and parameters that represent the distribution.
    /// 
    /// Subclasses must define this, but the arguments and return type vary between concrete classes.
    /// </summary>
    /// <param name="kwargs"></param>
    /// <returns></returns>
    (nn.Module, Parameter) ProbaDistributionNet(IDictionary<string, object> kwargs);
    /// <summary>
    /// Set parameters of the distribution.
    /// </summary>
    /// <param name="kwargs"></param>
    /// <returns> self </returns>
    IProbaWithParameter ProbaDistribution(IDictionary<string, object> kwargs);
}
