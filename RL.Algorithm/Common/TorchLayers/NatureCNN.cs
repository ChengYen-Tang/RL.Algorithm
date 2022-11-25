using System.Diagnostics;

namespace RL.Algorithm.Common.TorchLayers;

/// <summary>
/// CNN from DQN Nature paper:
///     Mnih, Volodymyr, et al.
///     "Human-level control through deep reinforcement learning."
///     Nature 518.7540 (2015): 529-533.
/// </summary>
public class NatureCNN : BaseFeaturesExtractor
{
    private readonly Sequential cnn;
    private readonly Sequential linear;

    /// <summary></summary>
    /// <param name="observationSpace"></param>
    /// <param name="featuresDim">
    /// Number of features extracted.
    /// This corresponds to the number of unit for the last layer.
    /// </param>
    public NatureCNN(Box observationSpace, int featuresDim = 512)
        : base(observationSpace, featuresDim)
    {
        Trace.Assert(Preprocessing.IsImageSpace(observationSpace, false),
            $"You should use NatureCNN only with images not with {observationSpace}\n" + "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n" + "If you are using a custom environment,\n" + "please check it using our env checker:\n");
        long nInputChannels = observationSpace.Shape[0];
        cnn = nn.Sequential(
            nn.Conv2d(nInputChannels, 32, 8, 4, 0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(),
            nn.Flatten()
            );

        long nFlatten;
        // Compute shape by doing one forward pass
        using (no_grad())
        {
            nFlatten = cnn.forward(as_tensor(observationSpace.Sample().AsFloatArray().ToList()).@float()).shape[1];
        }
        linear = nn.Sequential(nn.Linear(nFlatten, featuresDim), nn.ReLU());
    }

    public override Tensor Forward(Tensor observations)
        => linear.forward(cnn.forward(observations));
}
