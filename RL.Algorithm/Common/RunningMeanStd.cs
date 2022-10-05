namespace RL.Algorithm.Common;

/// <summary>
/// Calulates the running mean and std of a data stream
/// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
/// </summary>
internal class RunningMeanStd
{
    public ndarray Mean;
    public ndarray Var;
    private double count;

    /// <summary></summary>
    /// <param name="shape"> the shape of the data stream's output </param>
    /// <param name="epsilon"> helps with arithmetic issues </param>
    public RunningMeanStd(shape shape, double epsilon = 1e-4)
    {
        Mean = np.zeros(shape, np.Float64);
        Var = np.ones(shape, np.Float64);
        count = epsilon;
    }

    public RunningMeanStd Copy()
        => new(Mean.shape) { Mean = Mean.Copy(), Var = Var.Copy(), count = count };

    /// <summary>
    /// Combine stats from another ``RunningMeanStd`` object.
    /// </summary>
    /// <param name="other"> The other object to combine with. </param>
    public void Combine(RunningMeanStd other)
        => UpdateFromMoments(other.Mean, other.Var, other.count);

    public void Update(ndarray arr)
        => UpdateFromMoments(np.mean(arr, axis: 0), np.var(arr, axis: 0), arr.shape[0]);

    public void UpdateFromMoments(ndarray batchMean, ndarray batchVar, double batchCount)
    {
        ndarray delta = batchMean - Mean;
        double tot_count = count + batchCount;

        ndarray new_mean = Mean + delta * batchCount / tot_count;
        ndarray m_a = Var * count;
        ndarray m_b = batchVar * batchCount;
        ndarray m_2 = m_a + m_b + np.square(delta) * count * batchCount / (count + batchCount);
        ndarray new_var = m_2 / (count + batchCount);

        double new_count = batchCount + count;

        Mean = new_mean;
        Var = new_var;
        count = new_count;
    }
}
