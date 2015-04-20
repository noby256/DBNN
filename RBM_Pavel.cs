
#region iterate through batch
foreach (DataItem<double> dataItem in batch)
{
    //init visible layer states
    for (int i = 0; i < dataItem.Input.Length; i++)
    {
        visibleLayer.Neurons[i].LastState = dataItem.Input[i];
    }

    #region Gibbs sampling
    for (int k = 0; k <= _config.GibbsSamplingChainLength; k++)
    {
        //calculate hidden states probabilities
        hiddenLayer.Compute();

        #region accumulate negative phase
        if (k == _config.GibbsSamplingChainLength)
        {
            for (int i = 0; i < visibleLayer.Neurons.Length; i++)
            {
                for (int j = 0; j < hiddenLayer.Neurons.Length; j++)
                {
                    switch (_config.RBMDataType)
                    {
                        case RBMDataType.BinaryBinary:
                            nablaWeights[i, j] -= visibleLayer.Neurons[i].LastState *
                                                    hiddenLayer.Neurons[j].LastState;
                            break;
                        case RBMDataType.GaussianBinary:
                            nablaWeights[i, j] -= visibleLayer.Neurons[i].LastState*
                                                    hiddenLayer.Neurons[j].LastState;
                            break;
                    }


                    if (_config.RegularizationFactor > Double.Epsilon)
                    {
                        //regularization of weights
                        double regTerm = 0;
                        switch (_config.RegularizationType)
                        {
                            case RegularizationType.L1:
                                regTerm = _config.RegularizationFactor*
                                            Math.Sign(visibleLayer.Neurons[i].Weights[j]);
                                break;
                            case RegularizationType.L2:
                                regTerm = _config.RegularizationFactor*
                                            visibleLayer.Neurons[i].Weights[j];
                                break;
                        }
                        nablaWeights[i, j] -= regTerm;
                    }
                }
            }
            if (_config.UseBiases)
            {
                for (int i = 0; i < hiddenLayer.Neurons.Length; i++)
                {
                    nablaHiddenBiases[i] -= hiddenLayer.Neurons[i].LastState;
                }
                for (int i = 0; i < visibleLayer.Neurons.Length; i++)
                {
                    switch (_config.RBMDataType)
                    {
                        case RBMDataType.BinaryBinary:
                            nablaVisibleBiases[i] -= visibleLayer.Neurons[i].LastState;
                            break;
                        case RBMDataType.GaussianBinary:
                            nablaVisibleBiases[i] -= (visibleLayer.Neurons[i].LastState -
                                                        visibleLayer.Neurons[i].Bias);
                            break;
                    }
                }
            }

            break;
        }
        #endregion

        //sample hidden states
        for (int i = 0; i < hiddenLayer.Neurons.Length; i++)
        {
            hiddenLayer.Neurons[i].LastState = _r.NextDouble() <= hiddenLayer.Neurons[i].LastState ? 1d : 0d;
        }

        #region accumulate positive phase
        if (k == 0)
        {
            for (int i = 0; i < visibleLayer.Neurons.Length; i++)
            {
                for (int j = 0; j < hiddenLayer.Neurons.Length; j++)
                {
                    switch (_config.RBMDataType)
                    {
                        case RBMDataType.BinaryBinary:
                            nablaWeights[i, j] += visibleLayer.Neurons[i].LastState*
                                                    hiddenLayer.Neurons[j].LastState;
                            break;
                        case RBMDataType.GaussianBinary:
                            nablaWeights[i, j] += visibleLayer.Neurons[i].LastState*
                                                    hiddenLayer.Neurons[j].LastState;
                            break;
                    }
                }
            }
            if (_config.UseBiases)
            {
                for (int i = 0; i < hiddenLayer.Neurons.Length; i++)
                {
                    nablaHiddenBiases[i] += hiddenLayer.Neurons[i].LastState;
                }
                for (int i = 0; i < visibleLayer.Neurons.Length; i++)
                {
                    switch (_config.RBMDataType)
                    {
                        case RBMDataType.BinaryBinary:
                            nablaVisibleBiases[i] += visibleLayer.Neurons[i].LastState;
                            break;
                        case RBMDataType.GaussianBinary:
                            nablaVisibleBiases[i] += (visibleLayer.Neurons[i].LastState -
                                                        visibleLayer.Neurons[i].Bias);
                            break;
                    }
                }
            }
        }
        #endregion

        //calculate visible probs
        visibleLayer.Compute();

        //todo: may be not do sampling, like in 3.2 of http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
        //sample visible
        //for (int i = 0; i < visibleLayer.Neurons.Length; i++)
        //{
        //    visibleLayer.Neurons[i].LastState = _r.NextDouble() <= visibleLayer.Neurons[i].LastState ? 1d : 0d;
        //}

    }
    #endregion

}
#endregion

#region compute mean of wights nabla, and update them

for (int i = 0; i < visibleLayer.Neurons.Length; i++)
{
    for (int j = 0; j < hiddenLayer.Neurons.Length; j++)
    {
        lastMomentumSpeedWeights[i, j] = momentumSpeedWeights[i, j];
        momentumSpeedWeights[i, j] = _config.Momentum*momentumSpeedWeights[i, j] +
                                        nablaWeights[i, j]/batch.Count;
        visibleLayer.Neurons[i].Weights[j] += learningRate * localWeightGain[i, j] * momentumSpeedWeights[i, j];                            
                            
        hiddenLayer.Neurons[j].Weights[i] = visibleLayer.Neurons[i].Weights[j];
    }
}
if (_config.UseBiases)
{
    for (int i = 0; i < hiddenLayer.Neurons.Length; i++)
    {
        lastMomentumSpeedHiddenBiases[i] = momentumSpeedHiddenBiases[i];
        momentumSpeedHiddenBiases[i] = _config.Momentum*momentumSpeedHiddenBiases[i] +
                                        nablaHiddenBiases[i]/batch.Count;
        hiddenLayer.Neurons[i].Bias += learningRate * localBiasHiddelGain[i] * momentumSpeedHiddenBiases[i];
    }
    for (int i = 0; i < visibleLayer.Neurons.Length; i++)
    {
        lastMomentumSpeedVisibleBiases[i] = momentumSpeedVisibleBiases[i];
        momentumSpeedVisibleBiases[i] = _config.Momentum*momentumSpeedVisibleBiases[i] +
                                        nablaVisibleBiases[i]/batch.Count;
        visibleLayer.Neurons[i].Bias += learningRate * localBiasVisibleGain[i] * momentumSpeedVisibleBiases[i];
    }
}

#endregion