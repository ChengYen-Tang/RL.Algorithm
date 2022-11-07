namespace RL.Algorithm.Common.Distributions;

internal class StateDependentNoiseDistribution : Distribution, IProbaWithParameter
{
    private readonly int actionDim;
    private readonly bool useExpln;
    private readonly bool fullStd;
    private readonly double epsilon;
    private readonly bool learnFeatures;
    private readonly TanhBijector bijector;

    private int latentSdeDim;
    private Normal weightsDist = null!;
    private Tensor explorationMat = null!;
    private Tensor explorationMatrices = null!;
    private Tensor latentSde = null!;
    private Normal distribution = null!;

    public StateDependentNoiseDistribution(int actionDim, bool fullStd = true, bool useExpln = false, bool squashOutput = false, bool learnFeatures = false, double epsilon = 1e-6)
        => (this.actionDim, this.useExpln, this.fullStd, this.epsilon, this.learnFeatures, bijector)
        = (actionDim, useExpln, fullStd, epsilon, learnFeatures, squashOutput ? new(epsilon) : null!);

    public override Tensor ActionsFromParams(IDictionary<string, object> kwargs)
    {
        ProbaDistribution(kwargs);
        bool deterministic = kwargs.ContainsKey("deterministic") && (bool)kwargs["deterministic"];
        return GetActions(deterministic);
    }

    public override Tensor? Entropy()
    {
        if (bijector is not null)
            // No analytical form,
            // entropy needs to be estimated using -log_prob.mean()
            return null;
        return SumIndependentDims(distribution.entropy());
    }

    public Tensor GetNoise(Tensor latentSde)
    {
        latentSde = learnFeatures ? latentSde : latentSde.detach();
        // Default case: only one exploration matrix
        if (latentSde.shape[0] == 1 || latentSde.shape[0] != explorationMatrices.shape[0])
            return mm(latentSde, explorationMat);
        // Use batch matrix multiplication for efficient computation
        // (batch_size, n_features) -> (batch_size, 1, n_features)
        latentSde = latentSde.unsqueeze(dim: 1);
        // (batch_size, 1, n_actions)
        Tensor noise = bmm(latentSde, explorationMatrices);
        return noise.squeeze(dim: 1);
    }

    /// <summary>
    /// Get the standard deviation from the learned parameter
    /// (log of it by default). This ensures that the std is positive.
    /// </summary>
    /// <param name="logStd"></param>
    /// <returns></returns>
    public Tensor GetStd(Tensor logStd)
    {
        Tensor std;
        if (useExpln)
        {
            // From gSDE paper, it allows to keep variance
            // above zero and prevent it from growing too fast
            Tensor belowThreshold = exp(logStd) * (logStd <= 0);
            // Avoid NaN: zeros values that are below zero
            Tensor safeLogStd = logStd * (logStd > 0) + epsilon;
            Tensor aboveThreshold = (log1p(safeLogStd) + 1.0) * (logStd > 0);
            std = belowThreshold + aboveThreshold;
        }
        else
            // Use normal exponential
            std = exp(logStd);

        if (fullStd)
            return std;
        // Reduce the number of parameters:
        return ones(latentSdeDim, actionDim).to(logStd.device) * std;
    }

    public override Tensor LogProb(Tensor actions)
    {
        Tensor gaussianActions = bijector is not null ? TanhBijector.Inverse(actions) : actions;
        // log likelihood for a gaussian
        Tensor logProb = distribution.log_prob(gaussianActions);
        // Sum along action dim
        logProb = SumIndependentDims(logProb);

        if (bijector is not null)
            // Squash correction (from original SAC implementation)
            logProb -= sum(bijector.LogProbCorrection(gaussianActions), dim: 1);
        return logProb;
    }

    public override Tensor[] LogProbFromParams(IDictionary<string, object> kwargs)
    {
        Tensor action = ActionsFromParams(kwargs);
        Tensor logProb = LogProb(action);
        return new Tensor[] { action, logProb };
    }

    public override Tensor Mode()
    {
        Tensor actions = distribution.mean;
        if (bijector is not null)
            return TanhBijector.Forward(actions);
        return actions;
    }

    public IProbaWithParameter ProbaDistribution(IDictionary<string, object> kwargs)
    {
        Tensor latentSde = (kwargs["latent_sde"] as Tensor)!;
        this.latentSde = learnFeatures ? latentSde : latentSde.detach();
        Tensor logStd = (kwargs["log_std"] as Tensor)!;
        Tensor variance = mm(this.latentSde.pow(2), GetStd(logStd).pow(2));
        Tensor meanActions = (kwargs["mean_actions"] as Tensor)!;
        distribution = new Normal(meanActions, sqrt(variance + epsilon));
        return this;
    }

    public (nn.Module, Parameter) ProbaDistributionNet(IDictionary<string, object> kwargs)
    {
        // Network for the deterministic action, it represents the mean of the distribution
        int latentDim = (int)kwargs["latent_dim"];
        Linear meanActionsNet = nn.Linear(latentDim, actionDim);
        // When we learn features for the noise, the feature dimension
        // can be different between the policy and the noise network
        int? latentSdeDim = kwargs.ContainsKey("latent_sde_dim") ? (int)kwargs["latent_sde_dim"] : null;
        this.latentSdeDim = latentSdeDim ?? latentDim;
        // Reduce the number of parameters if needed
        Tensor logStd = fullStd ? ones(this.latentSdeDim, actionDim) : ones(this.latentSdeDim, 1);
        // Transform it to a parameter so it can be optimized
        float logStdInit = (float)kwargs["log_std_init"];
        logStd = nn.Parameter(logStd * logStdInit, requires_grad: true);
        // Sample an exploration matrix
        SampleWeights(logStd);
        return (meanActionsNet, (logStd as Parameter)!);
    }

    public override Tensor Sample()
    {
        Tensor noise = GetNoise(latentSde);
        Tensor actions = distribution.mean + noise;
        if (bijector is not null)
            TanhBijector.Forward(actions);
        return actions;
    }

    /// <summary>
    /// Sample weights for the noise exploration matrix,
    /// using a centered Gaussian distribution.
    /// </summary>
    /// <param name="logStd"></param>
    /// <param name="batchSize"></param>
    public void SampleWeights(Tensor logStd, int batchSize = 1)
    {
        Tensor std = GetStd(logStd);
        weightsDist = new(zeros_like(std), std);
        // Reparametrization trick to pass gradients
        explorationMat = weightsDist.rsample();
        // Pre-compute matrices in case of parallel exploration
        explorationMatrices = weightsDist.rsample(batchSize);
    }
}
