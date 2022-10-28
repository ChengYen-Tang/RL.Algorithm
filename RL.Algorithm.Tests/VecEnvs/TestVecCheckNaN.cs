namespace RL.Algorithm.Tests.VecEnvs;

[TestClass]
public class TestVecCheckNaN
{
    public VecEnv env = null!;

    [TestInitialize]
    public void Init()
    {
        env = new DummyVecEnv(new BaseEnv<DigitalSpace>[] { new NanAndInfEnv() });
        env = new VecCheckNan(env, raiseException: true);
        env.Step(np.array(new double[,] { { 0 } }));
    }

    [TestMethod]
    [ExpectedException(typeof(Error))]
    public void TestCheckActionNaN()
        => env.Step(np.array(new double[,] { { np.NaN } }));

    [TestMethod]
    [ExpectedException(typeof(Error))]
    public void TestCheckActionInf()
        => env.Step(np.array(new double[,] { { np.Inf } }));

    [TestMethod]
    [ExpectedException(typeof(Error))]
    public void TestCheckObservationInf()
        => env.Step(np.array(new double[,] { { -1 } }));

    [TestMethod]
    [ExpectedException(typeof(Error))]
    public void TestCheckObservationNaN()
        => env.Step(np.array(new double[,] { { 1 } }));

    [TestMethod]
    public void TestCheckSuccess()
    {
        env.Step(np.array(new double[,] { { 0, 1 }, { 0, 1 } }));
        env.Reset();
    }
}
