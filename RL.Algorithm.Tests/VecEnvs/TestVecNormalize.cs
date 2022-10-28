using RL.Algorithm.Common;

namespace RL.Algorithm.Tests.VecEnvs;

[TestClass]
public class TestVecNormalize
{
    [TestMethod]
    public void TestRunningMeanStd()
    {
        np.random rnd = new();
        foreach ((ndarray x1, ndarray x2, ndarray x3) in new[]
        { (rnd.randn(new(3)), rnd.randn(new(4)), rnd.randn(new(5))),
        (rnd.randn(new(3, 2)), rnd.randn(new(4, 2)), rnd.randn(new(5, 2))) })
        {
            RunningMeanStd rms = new(new(x1.shape.iDims[1..]), 0.0);
            ndarray xCat = np.concatenate((x1, x2, x3), 0);
            ndarray moments1 = np.stack(new[] { xCat.Mean(0), np.var(xCat, 0) });
            rms.Update(x1);
            rms.Update(x2);
            rms.Update(x3);
            ndarray moments2 = np.stack(new[] { rms.Mean, rms.Var });

            Assert.IsTrue(np.allclose(moments1, moments2));
        }
    }

    [TestMethod]
    public void TestCombiningStats()
    {
        np.random rnd = new();
        rnd.seed(4);

        foreach (shape shape in new shape[] { new(1), new(3), new(3, 4) })
        {
            List<ndarray> values = new();
            RunningMeanStd rms1 = new(shape);
            RunningMeanStd rms2 = new(shape);
            RunningMeanStd rms3 = new(shape);
            for (int i = 0; i < 15; i++)
            {
                ndarray value = rnd.randn(shape);
                rms1.Update(value);
                rms3.Update(value);
                values.Add(value);
            }
            for (int i = 0; i < 19; i++)
            {
                ndarray value = rnd.randn(shape) + 1.0;
                rms2.Update(value);
                rms3.Update(value);
                values.Add(value);
            }
            rms1.Combine(rms2);
            Assert.IsTrue(np.allclose(rms3.Mean, rms1.Mean));
            Assert.IsTrue(np.allclose(rms3.Var, rms1.Var));
            RunningMeanStd rms4 = rms3.Copy();
            Assert.IsTrue(np.allclose(rms4.Mean, rms3.Mean));
            Assert.IsTrue(np.allclose(rms4.Var, rms3.Var));
            Assert.IsTrue(np.allclose(rms4.Count, rms3.Count));
            Assert.AreNotEqual(rms4.Mean.GetHashCode(), rms3.Mean.GetHashCode());
            Assert.AreNotEqual(rms4.Var.GetHashCode(), rms3.Var.GetHashCode());
            ndarray xCat = np.concatenate(values, 0);
            Assert.IsTrue(np.allclose(np.mean(xCat, 0), rms4.Mean));
            Assert.IsTrue(np.allclose(np.var(xCat, 0), rms4.Var));
        }
    }

    [TestMethod]
    public void TestObsRmsVecNormalize()
    {
        DummyRewardEnv[] envs = new DummyRewardEnv[] { new(0), new(1) };
        VecEnv env = new DummyVecEnv(envs);
        VecNormalize normEnv = new(env);
        normEnv.Reset();
        Assert.IsTrue(np.allclose(normEnv.ObsRMS.Mean, 0.5, atol: 1e-4));
        Assert.IsTrue(np.allclose(normEnv.RetRMS.Mean, 0.0, atol: 1e-4));
        normEnv.Step(np.array(Enumerable.Range(0, envs.Length).Select(_ => normEnv.ActionSpace.Sample())));
        Assert.IsTrue(np.allclose(normEnv.ObsRMS.Mean, 1.25, atol: 1e-4));
        Assert.IsTrue(np.allclose(normEnv.RetRMS.Mean, 2, atol: 1e-4));
    }
}
