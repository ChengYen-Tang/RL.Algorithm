using System.Diagnostics;

namespace RL.Algorithm.Common.TorchLayers;

public abstract class BaseFeaturesExtractor : nn.Module
{
    protected readonly DigitalSpace observationSpace = null!;
    public readonly int featuresDim;

    protected BaseFeaturesExtractor(DigitalSpace observationSpace, int featuresDim = 0, string name = nameof(BaseFeaturesExtractor))
        : base(name)
    {
        Trace.Assert(featuresDim > 0);
        this.observationSpace = observationSpace;
        this.featuresDim = featuresDim;
    }

    public abstract Tensor Forward(Tensor observations);
}
