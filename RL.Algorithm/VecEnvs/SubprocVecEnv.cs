using BaseRLEnv.Spaces;
using BaseRLEnv;

namespace RL.Algorithm.VecEnvs;

/// <summary>
/// Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
/// process, allowing significant speed up when the environment is computationally complex.
/// For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
/// number of logical cores on your CPU.
/// </summary>
public class SubprocVecEnv : VecEnv
{
    public BaseEnv<DigitalSpace>[] Envs { get; init; }

    public SubprocVecEnv(BaseEnv<DigitalSpace>[] envs) : base(envs)
        => this.Envs = envs;

    public override ResetResult Reset(uint? seed = null, Dictionary<string, dynamic>? options = null)
    {
        var observations = new ndarray[NumEnvs];
        var infos = new Dictionary<string, dynamic>[NumEnvs];
        Parallel.For(0, NumEnvs, i => {
            BaseRLEnv.ResetResult reset = Envs[i].Reset(seed, options);
            observations[i] = reset.Observation;
            infos[i] = reset.Info;
        });
        return new ResetResult(np.stack(observations), infos);
    }

    public override StepResult Step(ndarray action)
    {
        var observations = new ndarray[NumEnvs];
        var rewards = new double[NumEnvs];
        var terminated = new bool[NumEnvs];
        var truncated = new bool[NumEnvs];
        var infos = new Dictionary<string, dynamic>[NumEnvs];
        Parallel.For(0, NumEnvs, i => {
            BaseRLEnv.StepResult step = Envs[i].Step((action[i] as ndarray)!);
            if (step.Terminated || step.Truncated)
            {
                step.Info["terminal_observation"] = step.Observation;
                Envs[i].Reset();
            }
            observations[i] = step.Observation;
            rewards[i] = step.Reward;
            terminated[i] = step.Terminated;
            truncated[i] = step.Truncated;
            infos[i] = step.Info;
        });
        return new StepResult(np.stack(observations), rewards, terminated, truncated, infos);
    }

    public override ndarray? Render(RanderMode randerMode)
    {
        List<ndarray> images = new();
        for (var i = 0; i < NumEnvs; i++)
        {
            ndarray? image = Envs[i].Render(randerMode);
            if (image is not null)
                images.Add(image);
        }
        return images.Count == NumEnvs ? np.stack(images.ToArray()) : null;
    }

    public override void Close()
    {
        Parallel.ForEach(Envs, env => {
            env.Close();
        });
    }
}
