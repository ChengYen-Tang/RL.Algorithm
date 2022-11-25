using System.Diagnostics;

namespace RL.Algorithm.Common.TorchLayers;

public static class TorchLayers
{
    /// <summary>
    /// Create a multi layer perceptron (MLP), which is
    /// a collection of fully-connected layers each followed by an activation function.
    /// </summary>
    /// <param name="inputDim"> Dimension of the input vector </param>
    /// <param name="outputDim"></param>
    /// <param name="netArch">
    /// Architecture of the neural net
    /// It represents the number of units per layer.
    /// The length of this list is the number of layers.
    /// </param>
    /// <param name="activationFn">
    /// The activation function
    /// to use after each layer.
    /// If is null, it will be nn.ReLU
    /// </param>
    /// <param name="squashOutput">
    /// Whether to squash the output using a Tanh activation function
    /// </param>
    /// <returns></returns>
    public static IEnumerable<nn.Module> CreateMlp(int inputDim, int outputDim, int[] netArch, nn.Module? activationFn = null, bool squashOutput = false)
    {
        activationFn ??= nn.ReLU();
        if (netArch.Length > 0)
        {
            yield return nn.Linear(inputDim, netArch[0]);
            yield return activationFn;
        }

        for (int index = 0; index < netArch.Length; index++)
        {
            yield return nn.Linear(netArch[index], netArch[index + 1]);
            yield return activationFn;
        }

        if (outputDim > 0)
        {
            int lastLayerDim = netArch.Length > 0 ? netArch[^1] : inputDim;
            yield return nn.Linear(lastLayerDim, outputDim);
        }
        if (squashOutput)
            yield return nn.Tanh();
    }

    /// <summary>
    /// Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).
    /// 
    /// The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    /// which can be different for the actor and the critic.
    /// It is assumed to be a list of ints or a dict.
    /// 
    /// 1. If it is a list, actor and critic networks will have the same architecture.
    ///     The architecture is represented by a list of integers (of arbitrary length (zero allowed))
    ///     each specifying the number of units per layer.
    ///     If the number of ints is zero, the network will be linear.
    /// 2. If it is a dict,  it should have the following structure:
    ///     ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
    ///     where the network architecture is a list as described in 1.
    ///     
    /// For example, to have actor and critic that share the same network architecture,
    /// you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).
    /// 
    /// If you want a different architecture for the actor and the critic,
    /// then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.
    /// 
    /// .. note::
    ///     Compared to their on-policy counterparts, no shared layers (other than the features extractor)
    ///     between the actor and the critic are allowed (to prevent issues with target networks).
    /// </summary>
    /// <param name="netArch"></param>
    /// <returns></returns>
    public static (ICollection<int>, ICollection<int>) GetActorCriticArch(Union<ICollection<int>, IDictionary<string, ICollection<int>>> netArch)
        => netArch.MatchFunc(
            (value) => (value, value),
            (value) =>
            {
                Trace.Assert(value.ContainsKey("pi"), "Error: no key 'pi' was provided in net_arch for the actor network");
                Trace.Assert(value.ContainsKey("qf"), "Error: no key 'qf' was provided in net_arch for the actor network");
                return (value["pi"], value["qf"]);
            });
}
