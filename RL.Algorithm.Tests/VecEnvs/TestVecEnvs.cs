namespace RL.Algorithm.Tests.VecEnvs;

[TestClass]
public class TestVecEnvs
{
    private const int nEnvs = 3;

    private static Type[] vecEnvClasses = new[] { typeof(DummyVecEnv), typeof(SubprocVecEnv) };

    private static ICollection<object?[]> VecEnvTestClassAndWrapper()
    {
        Type?[] vecEnvWrappers = new[] { null, typeof(VecNormalize), typeof(VecFrameStack) };

        List<object?[]> vecEnvs = new();
        foreach (Type vecEnvClass in vecEnvClasses)
            foreach (Type? vecEnvWrapperClass in vecEnvWrappers)
                vecEnvs.Add(new object?[] { vecEnvClass, vecEnvWrapperClass });
        return vecEnvs;
    }

    [DynamicData(nameof(VecEnvTestClassAndWrapper), DynamicDataSourceType.Method)]
    [TestMethod]
    public void TestVecEnvTerminalObs(Type vecEnvClass, Type? vecEnvWrapper)
    {
        int[] stepNums = Enumerable.Range(0, nEnvs).Select(value => value + 5).ToArray();
        VecEnv vecEnv = (Activator.CreateInstance(vecEnvClass, new object[] { stepNums.Select(value => new StepEnv(value)).ToArray() }) as VecEnv)!;

        if (vecEnvWrapper is not null)
            vecEnv = vecEnvWrapper == typeof(VecFrameStack)
                 ? (Activator.CreateInstance(vecEnvWrapper, new object[] { vecEnv, 2, ChannelsOrder.Default }) as VecEnv)!
                 : (Activator.CreateInstance(vecEnvWrapper, new object[] { vecEnv }) as VecEnv)!;

        ndarray zeroActs = np.zeros((nEnvs), np.Int32);
        ndarray prevObsB = vecEnv.Reset().Observation;
        for (int stepNum = 1; stepNum < stepNums.Max() + 1; stepNum++)
        {
            Algorithm.VecEnvs.StepResult result = vecEnv.Step(zeroActs);
            Assert.AreEqual(nEnvs, result.Observation.Count());
            Assert.AreEqual(nEnvs, result.Terminated.Length);
            Assert.AreEqual(nEnvs, result.Truncated.Length);
            Assert.AreEqual(nEnvs, result.Info.Length);

            var envIter = Extension.Enumerable.Zip(prevObsB, result.Observation, result.Terminated, result.Info, stepNums);
            foreach ((object prevObs, object obs, bool done, Dictionary<string, object> info, int finalStepNum) in envIter)
            {
                Assert.IsTrue(done == (stepNum == finalStepNum));
                if (!done)
                    Assert.IsFalse(info.ContainsKey("terminal_observation"));
                else
                {
                    ndarray terminalObs = (info["terminal_observation"] as ndarray)!;
                    Assert.IsTrue(np.allb(prevObs < terminalObs));
                    Assert.IsTrue(np.allb((obs as ndarray)! < (prevObs as ndarray)!));

                    if (vecEnv.GetType() != typeof(VecNormalize))
                    {
                        Assert.IsTrue(np.allb(((prevObs as ndarray)! + 1).Equals(terminalObs)));
                        Assert.IsTrue(np.allb((obs as ndarray)! == 0));
                    }
                }
            }
            prevObsB = result.Observation;
        }

        vecEnv.Close();
    }

    private static Dictionary<string, DigitalSpace> spaces = new()
    {
        { "discrete", new Discrete(2) },
        { "multidiscrete", new MultiDiscrete(np.array(new uint[]{ 2, 3 })) },
        { "multibinary", new MultiBinary(3) },
        { "continuous", new Box(np.zeros(2), np.ones(2)) }
    };

    private static ICollection<object[]> VecEnvTestClassAndSpace()
    {
        List<object[]> vecEnvs = new();
        foreach (var vecEnvClass in vecEnvClasses)
            foreach (DigitalSpace space in spaces.Values)
                vecEnvs.Add(new object[] { vecEnvClass, space });
        return vecEnvs;
    }

    [DynamicData(nameof(VecEnvTestClassAndSpace), DynamicDataSourceType.Method)]
    [TestMethod]
    public void TestVecEnvSingleSpace(Type vecEnvClass, DigitalSpace space)
    {
        void obsAssert(ndarray obs)
            => CheckVecEnvObs(obs, space);

        CheckVecEnvSpaces(vecEnvClass, space, obsAssert);
    }

    private static void CheckVecEnvSpaces(Type vecEnvClass, DigitalSpace space, Action<ndarray> obsAssert)
    {
        CustomGymEnv MakeEnv()
            => new(space);

        VecEnv vecEnv = (Activator.CreateInstance(vecEnvClass, new object[] { Enumerable.Range(0, nEnvs).Select(_ => MakeEnv()).ToArray() }) as VecEnv)!;
        Algorithm.VecEnvs.ResetResult resetResult = vecEnv.Reset();
        obsAssert(resetResult.Observation);

        bool[] dones = Enumerable.Range(0, nEnvs).Select(_ => false).ToArray();
        while (!dones.Any())
        {
            ndarray actions = np.stack(Enumerable.Range(0, nEnvs).Select(_ => vecEnv.ActionSpace.Sample()).ToArray());
            Algorithm.VecEnvs.StepResult stepResult = vecEnv.Step(actions);
            obsAssert(stepResult.Observation);
        }

        vecEnv.Close();
    }

    private static void CheckVecEnvObs(ndarray obs, Space space)
    {
        Assert.AreEqual((int)obs.shape[0], nEnvs);
        foreach (object value in obs)
            Assert.IsTrue(space.CheckConditions((value as ndarray)!).IsSuccess);
    }

    private CustomGymEnv MakeEnv(DigitalSpace space)
        => (Activator.CreateInstance(typeof(CustomGymEnv), new object[] { space }) as CustomGymEnv)!;
}
