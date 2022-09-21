using BaseRLEnv;
using BaseRLEnv.Spaces;
using NumpyDotNet;

namespace RL.Algorithm.VecEnvs;

public class DummyVecEnv : VecEnv
{
    public DummyVecEnv(BaseEnv<DigitalSpace>[] numEnvs) : base(numEnvs) { }

    public override ResetResult Reset(uint? seed = null, Dictionary<string, dynamic>? options = null)
    {
        var observations = new ndarray[NumEnvs.Length];
        var infos = new Dictionary<string, dynamic>[NumEnvs.Length];
        for (var i = 0; i < NumEnvs.Length; i++)
        {
            BaseRLEnv.ResetResult reset = NumEnvs[i].Reset(seed, options);
            observations[i] = reset.Observation;
            infos[i] = reset.Info;
        }
        return new ResetResult(np.stack(observations), infos);
    }

    public override StepResult Step(ndarray action)
    {
        var observations = new ndarray[NumEnvs.Length];
        var rewards = new double[NumEnvs.Length];
        var terminated = new bool[NumEnvs.Length];
        var truncated = new bool[NumEnvs.Length];
        var infos = new Dictionary<string, dynamic>[NumEnvs.Length];
        for (var i = 0; i < NumEnvs.Length; i++)
        {
            BaseRLEnv.StepResult step = NumEnvs[i].Step((action[i] as ndarray)!);
            observations[i] = step.Observation;
            rewards[i] = step.Reward;
            terminated[i] = step.Terminated;
            truncated[i] = step.Truncated;
            infos[i] = step.Info;
        }
        return new StepResult(np.stack(observations), rewards, terminated, truncated, infos);
    }

    public override ndarray? Render(RanderMode randerMode)
    {
        List<ndarray> images = new();
        for (var i = 0; i < NumEnvs.Length; i++)
        {
            ndarray? image  = NumEnvs[i].Render(randerMode);
            if (image is not null)
                images.Add(image);
        }
        return images.Count == NumEnvs.Length ? np.stack(images.ToArray()) : null;
    }

    public override void Close()
    {
        foreach (var env in NumEnvs)
            env.Close();
    }
}
