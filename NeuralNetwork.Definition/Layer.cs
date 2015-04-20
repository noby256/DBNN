using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Definition
{
    using System.Reflection;
    using System.Xml.Serialization;

    /// <summary>
    /// The layer.
    /// </summary>
    [Serializable]
    public class Layer
    {
        /// <summary>
        /// Gets or sets the count of neurons.
        /// </summary>
        public int CountOfNeurons { get; set; }

        /// <summary>
        /// Gets or sets the neurons.
        /// </summary>
        public Neuron[] Neurons { get; set; }

        /// <summary>
        /// Gets or sets the activation function.
        /// </summary>
        [XmlIgnore]
        public Func<double, Layer, double> ActivationFunction { get; set; }

        /// <summary>
        /// Gets the activation function name.
        /// </summary>
        public string ActivationFunctionName { get; set; }

        /// <summary>
        /// Gets the derivative activation function name.
        /// </summary>
        public string DerivativeActivationFunctionName { get; set; }

        /// <summary>
        /// Gets or sets the derivative activation function.
        /// </summary>
        [XmlIgnore]
        public Func<double, double> DerivativeActivationFunction { get; set; }
        private int _lengthSize;

        /// <summary>
        /// The r.
        /// </summary>
        private readonly Random r = new Random();

        /// <summary>
        /// The Notifier.
        /// </summary>
        private INotifier _notifier;

        /// <summary>
        /// Gets or sets a value indicating whether is binary.
        /// </summary>
        public bool IsBinary { get; set; }

        /// <summary>
        /// Initialise une nouvelle instance de la classe <see cref="Layer"/>.
        /// </summary>
        public Layer()
            : this(
                10,
                NeuroHelper.SigmoidalFunction,
                NeuroHelper.DerivativeSigmoidalFunction,
                0.01,
                new Program.ConsoleNotifier())
        {

        }

        /// <summary>
        /// Initialise une nouvelle instance de la classe <see cref="Layer"/>.
        /// </summary>
        /// <param name="countOfNeurons">
        /// The count of neurons.
        /// </param>
        /// <param name="activationFunction">
        /// The activation function.
        /// </param>
        /// <param name="derivativeActivationFunction">
        /// The derivative activation function.
        /// </param>
        /// <param name="alpha">
        /// The alpha.
        /// </param>
        /// <param name="notifier">
        /// The notifier.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// </exception>
        public Layer(int countOfNeurons,
            Func<double, Layer, double> activationFunction,
            Func<double, double> derivativeActivationFunction, double alpha, INotifier notifier)
        {
            if (activationFunction == null || derivativeActivationFunction == null)
            {
                throw new ArgumentNullException("Specify activation finctions!");
            }
            CountOfNeurons = countOfNeurons;
            Neurons = new Neuron[countOfNeurons];
            ActivationFunction = activationFunction;
            ActivationFunctionName = activationFunction.GetMethodInfo().Name;
            DerivativeActivationFunction = derivativeActivationFunction;
            DerivativeActivationFunctionName = derivativeActivationFunction.GetMethodInfo().Name;
            Alpha = alpha;
            _notifier = notifier;
        }

        /// <summary>
        /// Gets or sets the alpha.
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// The calculate outputs.
        /// </summary>
        /// <param name="values">
        /// The values.
        /// </param>
        /// <param name="layer">
        /// The layer.
        /// </param>
        public void CalculateOutputs(double[] values, Layer layer = null)
        {
            for (var i = 0; i < CountOfNeurons; i++)
            {
                Neurons[i].S = 0;
                for (var j = 0; j < values.Length; j++)
                {
                    Neurons[i].S += values[j] * Neurons[i].Weights[j];
                }
                Neurons[i].S -= Neurons[i].T;


                //  Neurons[i].Y = r.NextDouble() <= Neurons[i].Y ? 1d : 0d;
            }
            for (var i = 0; i < CountOfNeurons; i++)
            {
                Neurons[i].Y = ActivationFunction(Neurons[i].S, layer);
            }
        }

        /// <summary>
        /// The get outputs.
        /// </summary>
        /// <returns>
        /// The <see cref="double[]"/>.
        /// </returns>
        public double[] GetOutputs()
        {
            var result = new double[CountOfNeurons];
            for (var i = 0; i < Neurons.Length; i++)
            {
                result[i] = Neurons[i].Y;
            }
            return result;
        }

        /// <summary>
        /// The re calculate weights.
        /// </summary>
        /// <param name="prevLayer">
        /// The prev layer.
        /// </param>
        /// <param name="alpha">
        /// The alpha.
        /// </param>
        public void ReCalculateWeights(Layer prevLayer, double alpha)
        {
            for (var i = 0; i < CountOfNeurons; i++)
            {
                for (var j = 0; j < Neurons[i].Weights.Length; j++)
                {
                    Neurons[i].Weights[j] -= alpha * Neurons[i].Gamma * DerivativeActivationFunction(Neurons[i].Y) *
                                             prevLayer.Neurons[j].Y;
                }
                Neurons[i].T += alpha * Neurons[i].Gamma;
            }
        }

        /// <summary>
        /// The re calculate weights.
        /// </summary>
        /// <param name="y">
        /// The y.
        /// </param>
        /// <param name="alpha">
        /// The alpha.
        /// </param>
        public void ReCalculateWeights(double[] y, double alpha)
        {
            for (var i = 0; i < CountOfNeurons; i++)
            {
                for (var j = 0; j < Neurons[i].Weights.Length; j++)
                {
                    Neurons[i].Weights[j] -= alpha * Neurons[i].Gamma * DerivativeActivationFunction(Neurons[i].Y) *
                                             y[j];
                }
                Neurons[i].T += alpha * Neurons[i].Gamma;
            }
        }

        /// <summary>
        /// The initialize layer.
        /// </summary>
        /// <param name="countOfWeights">
        /// The count of weights.
        /// </param>
        /// <param name="sizesample">
        /// The sizesample.
        /// </param>
        public void InitializeLayer(int countOfWeights, int sizesample)
        {
            Neurons = new Neuron[CountOfNeurons];
            for (var i = 0; i < CountOfNeurons; i++)
            {
                Neurons[i] = new Neuron(countOfWeights, sizesample);
                Neurons[i].T = r.NextDouble() - 0.5;
                for (var j = 0; j < Neurons[i].CountOfWeights; j++)
                {
                    Neurons[i].Weights[j] = r.NextDouble() - 0.5;
                }
            }
        }

        /// <summary>
        /// The train as rbm.
        /// </summary>
        /// <param name="batch">
        /// The batch.
        /// </param>
        /// <param name="config">
        /// The config.
        /// </param>
        public void TrainAsRbm(BatchContainer batch, RbmConfig config)
        {
            var rbm = new Rbm(config);
            rbm.CountHiddenNeurons = CountOfNeurons;
            rbm.CountVisibleNeurons = Neurons[0].Weights.Length;
            var w = new double[rbm.CountVisibleNeurons, rbm.CountHiddenNeurons];
            batch.DataItems[0].ForEach(x => Console.WriteLine("Batch[0]={0}", x[0]));
            // Console.WriteLine("Batch[0]={0}",batch.DataItems[0]);
            for (int j = 0; j < Neurons.Length; j++)
            {
                for (int i = 0; i < Neurons[j].Weights.Length; i++)
                {
                    w[i, j] = Neurons[j].Weights[i];
                }
            }
            rbm.InitializeWeights();
            rbm.Train(batch, config.CountOfEpoches, _notifier);

            for (int j = 0; j < Neurons.Length; j++)
            {
                for (int i = 0; i < Neurons[j].Weights.Length; i++)
                {
                    Neurons[j].Weights[i] = rbm.W[i, j];
                }
            }
            for (int j = 0; j < Neurons.Length; j++)
            {
                Neurons[j].T = rbm.Tin[j];
            }

            var etalones = rbm.GetEtalones();

            var count = etalones.Count;
            Console.WriteLine("SIZE Sample:{0}", count);
            _lengthSize = count;
            for (int j = 0; j < Neurons.Length; j++)
            {
                Neurons[j].Yi = new double[count];
            }

            for (var i = 0; i < etalones.Count; i++)
            {
                for (int j = 0; j < Neurons.Length; j++)
                {
                    Neurons[j].Yi[i] = etalones[i][j];
                }
            }
        }

        /// <summary>
        /// The get batch container.
        /// </summary>
        /// <param name="batchSize">
        /// The batch size.
        /// </param>
        /// <returns>
        /// The <see cref="BatchContainer"/>.
        /// </returns>
        public BatchContainer GetBatchContainer(int batchSize)
        {
            var container = new BatchContainer(batchSize);
            var data = new List<double[]>();
            for (var i = 0; i < _lengthSize; i++)
            {
                var vector = new double[CountOfNeurons];
                for (var j = 0; j < Neurons.Length; j++)
                {
                    vector[j] = Neurons[j].Yi[i];
                }
                data.Add(vector);
                if ((i + 1) % batchSize == 0 || i == _lengthSize-1)
                {
                    container.DataItems.Add(data);
                    data = new List<double[]>();
                }
            }
            return container;
        }

    }
}
