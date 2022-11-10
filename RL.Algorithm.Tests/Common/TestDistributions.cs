using RL.Algorithm.Common.Distributions;
using TorchSharp;
using TorchSharp.Modules;

namespace RL.Algorithm.Tests.Common;

[TestClass]
public class TestDistributions
{
    private const int nActions = 2;
    private const int nFEATURES = 3;
    private readonly int nSamples = Convert.ToInt32(5e6);

    [TestMethod]
    public void TestBijector()
    {
        Tensor actions = ones(5) * 2.0;

        Tensor squashedActions = TanhBijector.Forward(actions);
        Assert.IsTrue(max(abs(squashedActions)).ToDouble() <= 1.0);
        Assert.IsTrue(TanhBijector.Inverse(squashedActions).isclose(actions).all().ToBoolean());
    }

    [TestMethod]
    public void TestSdeDistribution()
    {
        int nActions = 1;
        Tensor deterministicActions = ones(nSamples, nActions) * 0.1;
        Tensor state = ones(nSamples, nFEATURES) * 0.3;
        StateDependentNoiseDistribution dist = new(nActions, fullStd: true, squashOutput: false);

        random.manual_seed(1);
        (nn.Module _, Parameter logStd) = dist.ProbaDistributionNet(new Dictionary<string, object>() { { "latent_dim", nFEATURES } });
        dist.SampleWeights(logStd, nSamples);

        dist = (dist.ProbaDistribution(new Dictionary<string, object>() { { "mean_actions", deterministicActions }, { "log_std", logStd }, { "latent_sde", state } }) as StateDependentNoiseDistribution)!;
        Tensor actions = dist.GetActions();

        Assert.IsTrue(actions.mean().allclose(dist.Distribution.mean.mean(), rtol: 2e-3));
        Assert.IsTrue(actions.std().allclose(dist.Distribution.stddev.mean(), rtol: 2e-3));
    }

    public static ICollection<object[]> IProbaWithParameters()
    {
        return new object[][]
        {
            new object[] { new DiagGaussianDistribution(nActions) },
            new object[] { new StateDependentNoiseDistribution(nActions, squashOutput: false) }
        };
    }
    [DynamicData(nameof(IProbaWithParameters), DynamicDataSourceType.Method)]
    [TestMethod]
    public void TestEntropy(IProbaWithParameter dist)
    {
        random.manual_seed(1);
        Tensor deterministicActions = rand(1, nActions).repeat(nSamples, 1);
        (nn.Module _, Parameter logStd) = dist.ProbaDistributionNet(new Dictionary<string, object>() { { "latent_dim", nFEATURES }, { "log_std_init", log(tensor(0.2)) } });

        if (dist.GetType() == typeof(DiagGaussianDistribution))
            dist = dist.ProbaDistribution(new Dictionary<string, object>() { { "mean_actions", deterministicActions }, { "log_std", logStd } });
        else
        {
            Tensor state = rand(1, nFEATURES).repeat(nSamples, 1);
            (dist as StateDependentNoiseDistribution)?.SampleWeights(logStd, nSamples);
            dist = dist.ProbaDistribution(new Dictionary<string, object>() { { "mean_actions", deterministicActions }, { "log_std", logStd }, { "latent_sde", state } });
        }

        Tensor actions = dist.GetActions();
        Tensor entropy = dist.Entropy()!;
        Tensor logProb = dist.LogProb(actions);
        Assert.IsTrue(entropy.mean().allclose(-logProb.mean(), rtol: 5e-3));
    }

    public static ICollection<object[]> CategoricalParams()
    {
        long[] actionDims = new long[] { 2, 3 };
        return new object[][]
        {
            new object[] { new CategoricalDistribution(nActions), nActions },
            new object[] { new MultiCategoricalDistribution(actionDims), Convert.ToInt32(actionDims.Sum()) },
            new object[] { new BernoulliDistribution(nActions), nActions }
        };
    }
    [Ignore]
    [DynamicData(nameof(CategoricalParams), DynamicDataSourceType.Method)]
    [TestMethod]
    public void TestCategorical(IProba dist, int catActions)
    {
        random.manual_seed(1);
        Tensor actionLogits = rand(nSamples, catActions);
        dist = dist.ProbaDistribution(new Dictionary<string, object> { { "action_logits", actionLogits } });
        Tensor actions = dist.GetActions();
        Tensor entropy = dist.Entropy()!;
        Tensor logProb = dist.LogProb(actions);
        Assert.IsTrue(entropy.mean().allclose(-logProb.mean(), rtol: 5e-3));
    }
}
