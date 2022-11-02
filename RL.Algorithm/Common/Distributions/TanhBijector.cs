namespace RL.Algorithm.Common.Distributions;

/// <summary>
/// Bijective transformation of a probability distribution
/// using a squashing function(tanh)
/// TODO: use Pyro instead (https://pyro.ai/)
/// </summary>
internal class TanhBijector
{
    private readonly double epsilon;

    /// <summary></summary>
    /// <param name="epsilon"> small value to avoid NaN due to numerical imprecision. </param>
    public TanhBijector(double epsilon = 1e-6)
        => this.epsilon = epsilon;

    public static Tensor Forward(Tensor x)
        => tanh(x);

    /// <summary>
    /// Inverse of Tanh
    /// 
    /// Taken from Pyro: https://github.com/pyro-ppl/pyro
    /// 0.5 * torch.log((1 + x ) / (1 - x))
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor Atanh(Tensor x)
        => 0.5 * (x.log1p() - (-x).log1p());

    /// <summary>
    /// Inverse tanh.
    /// </summary>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor Inverse(Tensor y)
    {
        double eps = finfo(y.dtype).eps;
        return Atanh(y.clamp(min: -1.0 + eps, max: 1.0 - eps));
    }

    public Tensor LogProbCorrection(Tensor x)
        => log(1.0 - tanh(x).pow(2) + epsilon);
}
