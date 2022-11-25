namespace RL.Algorithm.Common.TorchLayers;

/// <summary>
/// Feature extract that flatten the input.
/// Used as a placeholder when feature extraction is not needed.
/// </summary>
public class FlattenExtractor : BaseFeaturesExtractor
{
    private readonly Flatten flatten;

    public FlattenExtractor(DigitalSpace observationSpace, string name = nameof(FlattenExtractor))
        : base(observationSpace, Convert.ToInt32(observationSpace.FlatDim()), name: name)
    => flatten = nn.Flatten();

    public override Tensor Forward(Tensor observations)
        => flatten(observations);
}
