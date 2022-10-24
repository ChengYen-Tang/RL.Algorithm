using System.Runtime.CompilerServices;

namespace RL.Algorithm.VecEnvs;

/// <summary>
/// Frame stacking wrapper for data.
/// 
/// Dimension to stack over is either first(channels-first) or
/// last(channels-last), which is detected automatically using
/// ``common.preprocessing.is_image_space_channels_first`` if
/// observation is an image space.
/// </summary>
public class StackedObservations
{
    public bool ChannelsFirst { get; init; }
    public int NStack { get; init; }

    private readonly int stackDimension;
    private readonly int repeatAxis;
    private ndarray stackedobs;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="numEnvs"> number of environments </param>
    /// <param name="nStack"> Number of frames to stack </param>
    /// <param name="observationSpace"> Environment observation space. </param>
    /// <param name="channelsOrder">
    /// If "first", stack on first image dimension. If "last", stack on last dimension.
    /// If Default, automatically detect channel to stack over in case of image observation or default to "last" (default).
    /// </param>
    public StackedObservations(int numEnvs, int nStack, Box observationSpace, ChannelsOrder channelsOrder = ChannelsOrder.Default)
    {
        NStack = nStack;
        (ChannelsFirst, stackDimension, stackedobs, repeatAxis) = ComputeStacking(numEnvs, nStack, observationSpace, channelsOrder);
    }

    /// <summary>
    /// Given an observation space, returns a new observation space with stacked observations
    /// </summary>
    /// <param name="observationSpace"></param>
    /// <returns> New observation space with stacked dimensions </returns>
    public Box StackObservationSpace(Box observationSpace)
    {
        ndarray low = np.repeat(observationSpace.Low, NStack, axis: repeatAxis);
        ndarray high = np.repeat(observationSpace.High, NStack, axis: repeatAxis);
        return new Box(low, high, low.shape, observationSpace.Type);
    }

    /// <summary>
    /// Resets the stackedobs, adds the reset observation to the stack, and returns the stack
    /// </summary>
    /// <param name="observation"> Reset observation </param>
    /// <returns> The stacked reset observation </returns>
    public ndarray Reset(ndarray observation)
    {
        stackedobs["..."] = 0;
        Update(observation);
        return stackedobs;
    }

    /// <summary>
    /// Adds the observations to the stack and uses the dones to update the infos.
    /// </summary>
    /// <param name="observation"> numpy array of observations </param>
    /// <param name="terminated"> bool array of terminated info </param>
    /// <param name="truncated"> bool array of truncated info </param>
    /// <param name="infos"> numpy array of info dicts </param>
    /// <returns> tuple of the stacked observations and the updated infos </returns>
    public (ndarray, Dictionary<string, object>[]) Update(ndarray observation, bool[] terminated, bool[] truncated, Dictionary<string, object>[] infos)
    {
        int stackAxSize = Convert.ToInt32(observation.shape[stackDimension]);
        stackedobs = np.roll(stackedobs, shift: -stackAxSize, axis: stackDimension);

        for (int i = 0; i < terminated.Length; i++)
        {
            if (terminated[i] || truncated[i])
            {
                if (infos[i].ContainsKey("terminal_observation"))
                {
                    ndarray oldTerminal = (infos[i]["terminal_observation"] as ndarray)!;
                    ndarray newTerminal = ChannelsFirst ? np.concatenate(new ndarray[] { (stackedobs[i, $":-{stackAxSize}", "..."] as ndarray)!, oldTerminal }, axis: 0)
                        : np.concatenate(new ndarray[] { (stackedobs[i, "...", new Slice(null, -stackAxSize, 1)] as ndarray)!, oldTerminal }, axis: stackDimension);
                    infos[i]["terminal_observation"] = newTerminal;
                }
                else
                    Log.Warning("VecFrameStack wrapping a VecEnv without terminal_observation info");
                stackedobs[i] = 0;
            }
        }

        Update(observation);
        return (stackedobs, infos);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Update(ndarray observation)
    {
        if (ChannelsFirst)
            stackedobs[":", $"-{observation.shape[stackDimension]} :", "..."] = observation;
        else
            stackedobs["...", $"-{observation.shape[stackDimension]} :"] = observation;
    }

    /// <summary>
    /// Calculates the parameters in order to stack observations
    /// </summary>
    /// <param name="numEnv"> Number of environments in the stack </param>
    /// <param name="nStack"> The number of observations to stack </param>
    /// <param name="observationSpace"> The observation space </param>
    /// <param name="channelsOrder"> The order of the channels </param>
    /// <returns> tuple of channels_first, stack_dimension, stackedobs, repeat_axis </returns>
    private static (bool, int, ndarray, int) ComputeStacking(int numEnv, int nStack, Box observationSpace, ChannelsOrder channelsOrder)
    {
        bool channelsFirst = (channelsOrder is ChannelsOrder.Default) ? Preprocessing.IsImageSpace(observationSpace) ? Preprocessing.IsImageSpaceChannelsFirst(observationSpace) : false : channelsOrder is ChannelsOrder.First;
        int stackDimension = channelsFirst ? 1 : -1;
        int repeatAxis = channelsFirst ? 0 : -1;
        ndarray low = np.repeat(observationSpace.Low, nStack, axis: repeatAxis);
        ndarray stackedobs = np.zeros(new shape(numEnv) + low.shape, low.Dtype);
        return (channelsFirst, stackDimension, stackedobs, repeatAxis);
    }

}

public enum ChannelsOrder
{
    First,
    Last,
    Default
}
