using BaseRLEnv.Spaces;

namespace RL.Algorithm;

internal static class Preprocessing
{
    /// <summary>
    /// Check if an image observation space (see ``is_image_space``)
    /// is channels-first(CxHxW, True) or channels-last(HxWxC, False).
    /// Use a heuristic that channel dimension is the smallest of the three.
    /// If second dimension is smallest, raise an exception(no support).
    /// </summary>
    /// <param name="observationSpace"></param>
    /// <returns> True if observation space is channels-first image, False if channels-last. </returns>
    public static bool IsImageSpaceChannelsFirst(Box observationSpace)
    {
        long smallestDimension = (long)np.argmin(np.array(observationSpace.Shape.iDims)).item();
        if (smallestDimension == 1)
            Log.Warning("Treating image space as channels-last, while second dimension was smallest of the three.");
        return smallestDimension == 0;
    }

    private static readonly long[] isImageSpace_Channels = new long[] { 1, 3, 4 };

    /// <summary>
    /// Check if a observation space has the shape, limits and dtype
    /// of a valid image.
    /// The check is conservative, so that it returns False if there is a doubt.
    /// 
    /// Valid images: RGB, RGBD, GrayScale with values in [0, 255]
    /// </summary>
    /// <param name="observationSpace"></param>
    /// <param name="checkChannels">
    /// Whether to do or not the check for the number of channels.
    /// e.g., with frame-stacking, the observation space may have more channels than expected.
    /// </param>
    /// <returns></returns>
    public static bool IsImageSpace(DigitalSpace observationSpace, bool checkChannels = false)
    {
        if (observationSpace.GetType() != typeof(Box) || observationSpace.Shape.iDims.Length != 3)
            return false;

        if (observationSpace.Type != np.UInt8)
            return false;

        if ((bool)np.any(observationSpace.Low != 0) || (bool)np.any(observationSpace.High != 255))
            return false;

        if (!checkChannels)
            return true;
        long nChannels = (IsImageSpaceChannelsFirst((observationSpace as Box)!)) ? observationSpace.Shape[0] : observationSpace.Shape[-1];
        return isImageSpace_Channels.Contains(nChannels);
    }
}
