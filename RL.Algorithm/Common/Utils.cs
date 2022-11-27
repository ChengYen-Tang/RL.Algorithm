using TorchSharp;

namespace RL.Algorithm.Common;

public static class Utils
{
    /// <summary>
    /// Retrieve PyTorch device.
    /// It checks that the requested device is available first.
    /// For now, it supports only cpu and cuda.
    /// By default, it tries to use the gpu.
    /// </summary>
    /// <param name="device"></param>
    /// <returns></returns>
    public static Device GetDevice(Union<Device, string> device)
    {
        Device thDevice = device.MatchFunc(
            (value) => value,
            (value) =>
            {
                if (value == "auto")
                    value = "cuda";
                return torch.device(value);
            });

        if (thDevice.type == DeviceType.CUDA && !cuda.is_available())
            return torch.device("cpu");
        return thDevice;
    }
}
