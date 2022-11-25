using MoreLinq.Extensions;
using TorchSharp;

namespace RL.Algorithm.Common.TorchLayers;

/// <summary>
/// Constructs an MLP that receives the output from a previous feature extractor (i.e. a CNN) or directly
/// the observations (if no feature extractor is applied) as an input and outputs a latent representation
/// for the policy and a value network.
/// The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
/// of them are shared between the policy network and the value network. It is assumed to be a list with the following
/// structure:
/// 
/// 1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
/// If the number of ints is zero, there will be no shared layers.
/// 2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
/// It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
/// If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.
/// 
/// For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
/// network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
/// would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
/// would be specified as [128, 128].
/// 
/// Adapted from Stable Baselines.
/// </summary>
public class MlpExtractor : nn.Module
{
    private readonly int lastLayerDimPi;
    private readonly int lastLayerDimVf;
    private readonly Sequential sharedNet = null!;
    private readonly Sequential policyNet = null!;
    private readonly Sequential valueNet = null!;

    /// <summary></summary>
    /// <param name="featureDim"> Dimension of the feature vector (can be the output of a CNN) </param>
    /// <param name="netArch">
    /// The specification of the policy and value networks.
    /// See above for details on its formatting.
    /// </param>
    /// <param name="activationFn">
    /// The activation function to use for the networks.
    /// </param>
    /// <param name="device"> default is auto </param>
    public MlpExtractor(int featureDim, ICollection<Union<int, IDictionary<string, ICollection<int>>>> netArch, nn.Module<Tensor, Tensor> activationFn, Union<Device, string>? device)
        : base(nameof(MlpExtractor))
    {
        Device thDevice = device == null ? Utils.GetDevice((Union<Device, string>)"auto") : Utils.GetDevice(device);
        List<nn.Module<Tensor, Tensor>> sharedNet = new();
        List<nn.Module<Tensor, Tensor>> policyNet = new();
        List<nn.Module<Tensor, Tensor>> valueNet = new();
        ICollection<int> policyOnlyLayers = Array.Empty<int>();
        ICollection<int> value_only_layers = Array.Empty<int>();
        int lastLayerDimShared = featureDim;
        foreach (Union<int, IDictionary<string, ICollection<int>>> layer in netArch)
        {
            bool needBreak = layer.MatchFunc(
                (value) =>
                {
                    sharedNet.Add(nn.Linear(lastLayerDimShared, value));
                    sharedNet.Add(activationFn);
                    lastLayerDimShared = value;
                    return false;
                },
                (value) =>
                {
                    if (value.ContainsKey("pi"))
                        policyOnlyLayers = value["pi"];
                    if (value.ContainsKey("vf"))
                        value_only_layers = value["vf"];
                    return true;
                });
            if (needBreak)
                break;
        }

        int lastLayerDimPi = lastLayerDimShared;
        int lastLayerDimVf = lastLayerDimShared;

        foreach ((int piLayerSize, int vfLayerSize) in policyOnlyLayers.ZipLongest(value_only_layers, (I1, I2) => Tuple.Create(I1, I2)))
        {
            if (piLayerSize > 0)
            {
                policyNet.Add(nn.Linear(lastLayerDimPi, piLayerSize));
                policyNet.Add(activationFn);
                lastLayerDimPi = piLayerSize;
            }
            if (vfLayerSize > 0)
            {
                valueNet.Add(nn.Linear(lastLayerDimVf, vfLayerSize));
                valueNet.Add(activationFn);
                lastLayerDimVf = vfLayerSize;
            }
        }

        this.lastLayerDimPi = lastLayerDimPi;
        this.lastLayerDimVf = lastLayerDimVf;

        this.sharedNet = nn.Sequential(sharedNet).to(thDevice);
        this.policyNet = nn.Sequential(policyNet).to(thDevice);
        this.valueNet = nn.Sequential(valueNet).to(thDevice);
    }

    /// <summary></summary>
    /// <param name="features"></param>
    /// <returns>
    /// latent_policy, latent_value of the specified network.
    /// If all layers are shared, then ``latent_policy == latent_value``
    /// </returns>
    public (Tensor, Tensor) Forward(Tensor features)
    {
        Tensor sharedLatent = sharedNet.forward(features);
        return (policyNet.forward(sharedLatent), valueNet.forward(sharedLatent));
    }

    public Tensor ForwardActor(Tensor features)
        => policyNet.forward(sharedNet.forward(features));

    public Tensor ForwardCritic(Tensor features)
        => valueNet.forward(sharedNet.forward(features));
}
