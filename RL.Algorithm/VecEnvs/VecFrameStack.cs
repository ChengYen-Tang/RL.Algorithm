using BaseRLEnv;
using BaseRLEnv.Spaces;

namespace RL.Algorithm.VecEnvs;

/// <summary>
/// Frame stacking wrapper for vectorized environment. Designed for image observations.
/// Uses the StackedObservations class, or StackedDictObservations depending on the observations space
/// </summary>
public class VecFrameStack : VecEnvWrapper
{
    private readonly StackedObservations stackedObservations;

    /// <summary></summary>
    /// <param name="vEnv"> the vectorized environment to wrap </param>
    /// <param name="nStack"> Number of frames to stack </param>
    /// <param name="channelsOrder">
    /// If "first", stack on first image dimension. If "last", stack on last dimension.
    /// If Default, automatically detect channel to stack over in case of image observation or default to "last" (default).
    /// Alternatively channels_order can be a dictionary which can be used with environments with Dict observation spaces
    /// </param>
    /// <exception cref="Error"> VecFrameStack only works with gym.spaces.Box observation spaces </exception>
    public VecFrameStack(VecEnv vEnv, int nStack, ChannelsOrder channelsOrder = ChannelsOrder.Default)
        : base (vEnv)
    {
        if (vEnv.ObservationSpace.GetType() != typeof(Box))
            throw new Error("VecFrameStack only works with gym.spaces.Box observation spaces");
        Box wrappedObsSpace = (vEnv.ObservationSpace as Box)!;
        stackedObservations = new(vEnv.NumEnvs, nStack, wrappedObsSpace, channelsOrder);
        ObservationSpace = stackedObservations.StackObservationSpace(wrappedObsSpace);
    }

    public override ResetResult Reset(uint? seed = null, Dictionary<string, dynamic>? options = null)
    {
        ResetResult result = VEnv.Reset(seed, options);
        ndarray observation = stackedObservations.Reset(result.Observation);
        return new(observation, result.Info);
    }

    public override StepResult Step(ndarray action)
    {
        StepResult result = VEnv.Step(action);
        (ndarray observations, Dictionary<string, dynamic>[] infos) = stackedObservations.Update(result.Observation, result.Terminated, result.Truncated, result.Info);
        return new(observations, result.Reward, result.Terminated, result.Truncated, infos);
    }

    public override void Close()
        => VEnv.Close();

    public override ndarray? Render(RanderMode randerMode)
        => VEnv.Render(randerMode);
}
