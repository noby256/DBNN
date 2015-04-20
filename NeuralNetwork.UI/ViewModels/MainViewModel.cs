using System;
using System.Linq;
using System.Windows.Data;
using NeuralNetwork.Definition;
using NeuralNetwork.Models;

namespace NeuralNetwork.UI.ViewModels
{
    using System.Threading;

    public class MainViewModel : BaseViewModel
	{
        /// <summary>
        /// Gets or sets a value indicating whether is enabled.
        /// </summary>
        public bool IsEnabled { get; set; }

        /// <summary>
        /// Gets or sets the emin.
        /// </summary>
        public double Emin { get; set; }

        /// <summary>
        /// Gets or sets the alpha.
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// Gets or sets the batch size.
        /// </summary>
        public int BatchSize { get; set; }

        /// <summary>
        /// Gets or sets the functions.
        /// </summary>
        public ListCollectionView Functions { get; set; }

        /// <summary>
        /// Gets or sets the neural network.
        /// </summary>
        public NNetwork NeuralNetwork { get; set; }

        /// <summary>
        /// Initialise une nouvelle instance de la classe <see cref="MainViewModel"/>.
        /// </summary>
        public MainViewModel()
		{
			Emin = 1e-4;
			Alpha = 0.01;
			BatchSize = 10;
		}

        /// <summary>
        /// The create train network.
        /// </summary>
        /// <param name="network">
        /// The network.
        /// </param>
        /// <param name="batch">
        /// The batch.
        /// </param>
        /// <param name="notifier">
        /// The notifier.
        /// </param>
        /// <param name="token">
        /// The token.
        /// </param>
        public void CreateTrainNetwork(NeuralNetworkModel network, BatchContainer batch, INotifier notifier, CancellationToken token)
		{
			var batches = new BatchContainer(batch.BatchSize);
            NeuralNetwork = new NNetwork(network.CountOfLayers, notifier);
			for (var i = 0; i < network.CountOfLayers; i++)
			{
                NeuralNetwork.Layers[i] = new Layer(network.Layers[i].CountOfNeurons,
					NeuroHelper.SetActivationFunction(network.Layers[i].ActivationFunction),
					NeuroHelper.SetDerivativeActivationFunction(network.Layers[i].ActivationFunction),
					network.Alpha, notifier);
                NeuralNetwork.Layers[i].InitializeLayer(network.Layers[i].WeightsCount, network.TrainingSetCount);
				if (!network.Layers[i].PreTrain) continue;
				var config = new RbmConfig
					             {
						             Algorithm = network.Layers[i].RbmConfiguration.Algorithm,
						             CountOfEpoches = network.Layers[i].RbmConfiguration.CountOfEpoches,
						             Alpha = network.Layers[i].RbmConfiguration.RbmAlpha,
						             K = network.Layers[i].RbmConfiguration.KParameter,
						             LoggingEnabled = network.Layers[i].RbmConfiguration.LoggingEnabled,
						             InputDataType = network.Layers[i].RbmConfiguration.InputDataType
					             };
				if (i == 0)
				{
                    NeuralNetwork.Layers[i].TrainAsRbm(batch, config);
				}
				else
				{
					batches = NeuralNetwork.Layers[i - 1].GetBatchContainer(batch.BatchSize);
                    NeuralNetwork.Layers[i].TrainAsRbm(batches, config);
				}
			}
            NeuralNetwork.Alpha = Alpha;
            NeuralNetwork.Train(batch, token);
		}
	}
}
