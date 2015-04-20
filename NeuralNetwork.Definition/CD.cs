using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Definition
{
    [Serializable]
    internal class ContrastiveDivergence : ILearningStrategy<IMultilayerNeuralNetwork>
    {

        private LearningAlgorithmConfig _config = null;
        private Random _r = null;


        internal ContrastiveDivergence(LearningAlgorithmConfig config)
        {
            _config = config;
            _r = new Random(Helper.GetSeed());
        }


        public void Train(IMultilayerNeuralNetwork network, IList<DataItem<double>> data)
        {
            Logger.Instance.Log("Contrastive Divergence starts...");

            if (_config.BatchSize == -1)
            {
                _config.BatchSize = data.Count;
            }

            ILayer visibleLayer = network.Layers[0];
            ILayer hiddenLayer = network.Layers[1];

            if (!_config.UseBiases)
            {
                for (int i = 0; i < visibleLayer.Neurons.Length; i++)
                {
                    visibleLayer.Neurons[i].Bias = 0d;
                }
                for (int i = 0; i < hiddenLayer.Neurons.Length; i++)
                {
                    hiddenLayer.Neurons[i].Bias = 0d;
                }
            }

            //init momentum
            double[,] momentumSpeedWeights = new double[visibleLayer.Neurons.Length, hiddenLayer.Neurons.Length];
            double[] momentumSpeedVisibleBiases = new double[visibleLayer.Neurons.Length];
            double[] momentumSpeedHiddenBiases = new double[hiddenLayer.Neurons.Length];

            //init stop factors
            bool stopFlag = false;
            double lastError = Double.MaxValue;
            double lastErrorChange = double.MaxValue;
            double learningRate = _config.LearningRate;

            int currentEpoche = 0;
            BatchEnumerator<DataItem<double>> batchEnumerator = new BatchEnumerator<DataItem<double>>(data, _config.BatchSize, true);
            do
            {

                DateTime dtStart = DateTime.Now;


                //start batch processing
                foreach (IList<DataItem<double>> batch in batchEnumerator)
                {

                    //batch gradient
                    double[,] nablaWeights = new double[visibleLayer.Neurons.Length, hiddenLayer.Neurons.Length];
                    double[] nablaHiddenBiases = new double[hiddenLayer.Neurons.Length];
                    double[] nablaVisibleBiases = new double[visibleLayer.Neurons.Length]; //todo: recheck all chain of calculations


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
                                        nablaWeights[i, j] -= visibleLayer.Neurons[i].LastState *
                                                             hiddenLayer.Neurons[j].LastState;
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
                                        nablaVisibleBiases[i] -= visibleLayer.Neurons[i].LastState;
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
                                        nablaWeights[i, j] += visibleLayer.Neurons[i].LastState *
                                                             hiddenLayer.Neurons[j].LastState;
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
                                        nablaVisibleBiases[i] += visibleLayer.Neurons[i].LastState;
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
                            momentumSpeedWeights[i, j] = _config.Momentum * momentumSpeedWeights[i, j] +
                                                         nablaWeights[i, j] / batch.Count;
                            visibleLayer.Neurons[i].Weights[j] += learningRate * momentumSpeedWeights[i, j];
                            hiddenLayer.Neurons[j].Weights[i] = visibleLayer.Neurons[i].Weights[j];
                        }
                    }
                    if (_config.UseBiases)
                    {
                        for (int i = 0; i < hiddenLayer.Neurons.Length; i++)
                        {
                            momentumSpeedHiddenBiases[i] = _config.Momentum * momentumSpeedHiddenBiases[i] +
                                                           nablaHiddenBiases[i] / batch.Count;
                            hiddenLayer.Neurons[i].Bias += learningRate * momentumSpeedHiddenBiases[i];
                        }
                        for (int i = 0; i < visibleLayer.Neurons.Length; i++)
                        {
                            momentumSpeedVisibleBiases[i] = _config.Momentum * momentumSpeedVisibleBiases[i] +
                                                            nablaVisibleBiases[i] / batch.Count;
                            visibleLayer.Neurons[i].Bias += learningRate * momentumSpeedVisibleBiases[i];
                        }
                    }

                    #endregion

                }


                #region Logging and error calculation
                string msg = "Epoche #" + currentEpoche;

                #region calculate error

                if (currentEpoche % _config.CostFunctionRecalculationStep == 0)
                {

                    #region calculatin energy
                    //double energy = 0;
                    //foreach (DataItem<double> dataItem in data)
                    //{
                    //    double[] hidStates = hiddenLayer.Compute(dataItem.Input);
                    //    for (int i = 0; i < visibleLayer.Neurons.Length; i++)
                    //    {
                    //        energy -= visibleLayer.Neurons[i].Bias*dataItem.Input[i];
                    //        for (int j = 0; j < hiddenLayer.Neurons.Length; j++)
                    //        {
                    //            energy -= dataItem.Input[i]*hidStates[j]*visibleLayer.Neurons[i].Weights[j];
                    //        }
                    //    }
                    //    for (int i = 0; i < hiddenLayer.Neurons.Length; i++)
                    //    {
                    //        energy -= hidStates[i]*hiddenLayer.Neurons[i].Bias;
                    //    }
                    //}
                    //energy /= data.Count;
                    //msg += "; Enegry is " + energy;
                    #endregion

                    #region calculating squared error with reconstruction

                    IMetrics<double> sed = MetricsCreator.SquareEuclideanDistance();
                    double d = 0;
                    foreach (DataItem<double> dataItem in data)
                    {
                        d += sed.Calculate(dataItem.Input, network.ComputeOutput(dataItem.Input));
                    }
                    msg += "; SqDist is " + d;

                    lastErrorChange = Math.Abs(lastError - d);
                    lastError = d;

                    #endregion

                }

                #endregion

                msg += "; Time: " + (DateTime.Now - dtStart).Duration().ToString();
                Logger.Instance.Log(msg);
                #endregion

                #region stop condition

                currentEpoche++;
                if (currentEpoche >= _config.MaxEpoches)
                {
                    stopFlag = true;
                    Logger.Instance.Log("Stop: currentEpoche:" + currentEpoche + " >= _config.MaxEpoches:" + _config.MaxEpoches);
                }
                else if (_config.MinError >= lastError)
                {
                    stopFlag = true;
                    Logger.Instance.Log("Stop: _config.MinError:" + _config.MinError + " >= lastError:" + lastError);
                }
                else if (_config.MinErrorChange >= lastErrorChange)
                {
                    stopFlag = true;
                    Logger.Instance.Log("Stop: _config.MinErrorChange:" + _config.MinErrorChange + " >= lastErrorChange:" + lastErrorChange);
                }

                #endregion

            } while (!stopFlag);


        }
    }
}
