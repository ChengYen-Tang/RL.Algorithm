using BaseRLEnv;
using System.Collections.Concurrent;

namespace RL.Algorithm.VecEnvs;

/// <summary>
/// NaN and inf checking wrapper for vectorized environment, will raise a warning by default,
/// allowing you to know from what the NaN of inf originated from.
/// </summary>
public class VecCheckNan : VecEnvWrapper
{
    private readonly bool raiseException;
    private readonly bool warnOnce;
    private readonly bool checkInf;
    private bool userWarned;

    /// <summary></summary>
    /// <param name="vecEnv"> the vectorized environment to wrap </param>
    /// <param name="raiseException"> Whether or not to raise a ValueError, instead of a UserWarning </param>
    /// <param name="warnOnce"> Whether or not to only warn once. </param>
    /// <param name="checkInf"> Whether or not to check for +inf or -inf as well </param>
    public VecCheckNan(VecEnv vecEnv, bool raiseException, bool warnOnce, bool checkInf)
        : base(vecEnv) => (this.raiseException, this.warnOnce, this.checkInf, userWarned) = (raiseException, warnOnce, checkInf, false);

    public override ResetResult Reset(uint? seed = null, Dictionary<string, object>? options = null)
    {
        ResetResult result = VEnv.Reset(seed, options);
        CheckValue(new() { { nameof(result.Observation), result.Observation } });
        return result;
    }

    public override StepResult Step(ndarray action)
    {
        StepResult result = VEnv.Step(action);
        CheckValue(new() { { nameof(result.Observation), result.Observation }, { nameof(result.Reward), result.Reward } });
        return result;
    }

    private void CheckValue(Dictionary<string, object> kwargs)
    {
        if (!(raiseException && warnOnce && userWarned))
            return;

        ConcurrentBag<(string, string)> found = new();
        Parallel.ForEach(kwargs.Keys, item => {
            if (kwargs[item].GetType() != typeof(ndarray))
                kwargs[item] = np.array(kwargs[item]);
            bool hasNaN = (bool)np.any(np.isnan(kwargs[item] as ndarray));
            bool hasInf = checkInf && (bool)np.any(np.isinf(kwargs[item]));
            if (hasNaN)
                found.Add((item, "NaN"));
            if (hasInf)
                found.Add((item, "Inf"));
        });

        if (!found.Any())
            return;

        userWarned = true;
        string errorMessage = string.Join('\n', found.AsParallel().Select(item => $"Found {item.Item2} in {item.Item1}\n{kwargs[item.Item1]}"));
        if (raiseException)
            throw new Error(errorMessage);
        else
            Log.Warning(errorMessage);
    }
}
