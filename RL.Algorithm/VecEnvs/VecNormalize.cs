namespace RL.Algorithm.VecEnvs;

/// <summary>
/// A moving average, normalizing wrapper for vectorized environment.
/// has support for saving/loading moving average,
/// </summary>
public class VecNormalize : VecEnvWrapper
{
    private readonly bool normObs;
    private readonly bool normReward;
    private readonly bool training;
    private readonly double clipObs;
    private readonly double clipReward;
    private readonly double gamma;
    private readonly double epsilon;
    private readonly RunningMeanStd retRMS;
    private readonly RunningMeanStd obsRMS;

    public ndarray Returns { get; private set; }
    public ndarray OldObs { get; private set; }
    public double[] OldReward { get; private set; }

    /// <summary></summary>
    /// <param name="venv"> the vectorized environment to wrap </param>
    /// <param name="training"> Whether to update or not the moving average </param>
    /// <param name="normObs"> Whether to normalize observation or not (default: True) </param>
    /// <param name="normReward"> Whether to normalize rewards or not (default: True) </param>
    /// <param name="clipObs"> Max absolute value for observation </param>
    /// <param name="clipReward"> Max value absolute for discounted reward </param>
    /// <param name="gamma"> discount factor </param>
    /// <param name="epsilon"> To avoid division by zero </param>
    public VecNormalize(VecEnv venv, bool training = true, bool normObs = true, bool normReward = true,
        double clipObs = 10, double clipReward = 10, double gamma = 0.99, double epsilon = 1e-8)
        :base(venv)
    {
        this.normObs = normObs;

        obsRMS = new(ObservationSpace.Shape);
        retRMS = new(new(Array.Empty<int>()));
        this.clipObs = clipObs;
        this.clipReward = clipReward;

        Returns = np.zeros(NumEnvs);
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.training = training;
        this.normReward = normReward;
        OldObs = new();
        OldReward = Array.Empty<double>();
    }

    public override ResetResult Reset(uint? seed = null, Dictionary<string, object>? options = null)
    {
        ResetResult result = VEnv.Reset();
        OldObs = result.Observation;
        Returns = np.zeros(NumEnvs);
        if (training && normObs)
            obsRMS.Update(result.Observation);
        return new(NormalizeObs(result.Observation), result.Info);
    }

    public override StepResult Step(ndarray action)
    {
        StepResult result = VEnv.Step(action);
        OldObs = result.Observation;
        OldReward = result.Reward;

        if (training && normObs)
            obsRMS.Update(result.Observation);

        if (training)
            UpdateReward(result.Reward);

        bool[] dones = new bool[NumEnvs];
        Parallel.For(0, NumEnvs, i => {
            if (result.Terminated[i] || result.Truncated[i])
            {
                result.Info[i]["terminal_observation"] = NormalizeObs((result.Info[i]["terminal_observation"] as ndarray)!);
                dones[i] = true;
            }
            else
                dones[i] = false;
        });

        Returns[dones] = 0;
        return new(NormalizeObs(result.Observation), NormalizeReward(result.Reward), result.Terminated, result.Truncated, result.Info);
    }

    /// <summary>
    /// Normalize observations using this VecNormalize's observations statistics.
    /// Calling this method does not update statistics.
    /// </summary>
    /// <param name="obs"></param>
    /// <returns></returns>
    public ndarray NormalizeObs(ndarray obs)
    {
        ndarray obsTemp = obs.Copy();
        if (normObs)
            obsTemp = NormalizeObs(obs, obsRMS).astype(np.Float32);
        return obsTemp;
    }

    /// <summary>
    /// Normalize rewards using this VecNormalize's rewards statistics.
    /// Calling this method does not update statistics.
    /// </summary>
    /// <param name="reward"></param>
    /// <returns></returns>
    public double[] NormalizeReward(double[] reward)
    {
        if (normReward)
            reward = np.clip(reward / np.sqrt(retRMS.Var + epsilon), -clipReward, clipReward).AsDoubleArray();
        return reward;
    }

    private ndarray NormalizeObs(ndarray obs, RunningMeanStd obsRMS)
        => np.clip(((obs - obsRMS.Mean) / np.sqrt(obsRMS.Var + epsilon)) as ndarray, -clipObs, clipObs);

    private void UpdateReward(double[] reward)
    {
        Returns = Returns * gamma + np.array(reward);
        retRMS.Update(Returns);
    }
}
