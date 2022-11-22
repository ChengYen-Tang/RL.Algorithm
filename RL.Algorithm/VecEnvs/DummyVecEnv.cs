namespace RL.Algorithm.VecEnvs;

/// <summary>
/// Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
/// Python process.This is useful for computationally simple environment such as ``cartpole-v1``,
/// as the overhead of multiprocess or multithread outweighs the environment computation time.
/// This can also be used for RL methods that
/// require a vectorized environment, but that you want a single environments to train with.
/// </summary>
public class DummyVecEnv : VecEnv
{
    public DigitalEnv[] Envs { get; init; }

    public DummyVecEnv(DigitalEnv[] envs) : base(envs)
        => this.Envs = envs;

    public override ResetResult Reset(uint? seed = null, Dictionary<string, object>? options = null)
    {
        var observations = new ndarray[NumEnvs];
        var infos = new Dictionary<string, object>[NumEnvs];
        for (var i = 0; i < NumEnvs; i++)
        {
            Env.ResetResult reset = Envs[i].Reset(seed, options);
            observations[i] = reset.Observation;
            infos[i] = reset.Info;
        }
        return new ResetResult(np.stack(observations), infos);
    }

    public override StepResult Step(ndarray action)
    {
        var observations = new ndarray[NumEnvs];
        var rewards = new double[NumEnvs];
        var terminated = new bool[NumEnvs];
        var truncated = new bool[NumEnvs];
        var infos = new Dictionary<string, object>[NumEnvs];
        for (int i = 0; i < NumEnvs; i++)
        {
            Env.StepResult step = Envs[i].Step((action[i] as ndarray)!);
            rewards[i] = step.Reward;
            terminated[i] = step.Terminated;
            truncated[i] = step.Truncated;
            if (step.Terminated || step.Truncated)
            {
                Env.ResetResult reset = Envs[i].Reset();
                observations[i] = reset.Observation;
                infos[i] = reset.Info;
                infos[i]["terminal_observation"] = step.Observation;
            }
            else
            {
                observations[i] = step.Observation;
                infos[i] = step.Info;
            }
        }
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
        foreach (var env in Envs)
            env.Close();
    }
}
