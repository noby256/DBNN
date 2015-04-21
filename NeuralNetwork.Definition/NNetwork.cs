using System;
using System.IO;
using System.Linq;
using System.Xml.Serialization;
using NLog;

namespace NeuralNetwork.Definition
{
    using System.Threading;

    /// <summary>
    /// The n network.
    /// </summary>
    [Serializable]
    [XmlRoot("NNetwork", Namespace = "")]
    public class NNetwork
    {
        /// <summary>
        /// The log.
        /// </summary>
        private static Logger log;

        /// <summary>
        /// Gets or sets the count of layers.
        /// </summary>
        [XmlElement("CountOfLayers")]
        public int CountOfLayers { get; set; }
        /// <summary>
        /// Gets or sets the layers.
        /// </summary>
        public Layer[] Layers { get; set; }

        /// <summary>
        /// Gets or sets the alpha.
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// Gets or sets the alpha max.
        /// </summary>
        public double AlphaMax { get; set; }

        /// <summary>
        /// Gets or sets the emin.
        /// </summary>
        public double Emin { get; set; }

        /// <summary>
        /// Gets or sets the e emin.
        /// </summary>
        public double EEmin { get; set; }

        /// <summary>
        /// Gets or sets the training set count.
        /// </summary>
        public int TrainingSetCount { get; set; }

        /// <summary>
        /// Gets or sets the iterations count.
        /// </summary>
        public int IterationsCount { get; set; }

        /// <summary>
        /// The r.
        /// </summary>
        private readonly Random r = new Random();

        /// <summary>
        /// The Notifier.
        /// </summary>
        [XmlIgnore]
        public INotifier Notifier;

        /// <summary>
        /// Initialise une nouvelle instance de la classe <see cref="NNetwork"/>.
        /// </summary>
        /// <param name="countOfLayers">
        /// The count of layers.
        /// </param>
        /// <param name="notifier">
        /// The notifier.
        /// </param>
        public NNetwork(int countOfLayers, INotifier notifier)
            : this(countOfLayers, 0.1, 0.4, 0.0001, 0.0005)
        {
            Notifier = notifier;
            CountOfLayers = countOfLayers;
        }

        public NNetwork()
            : this(0, 0.1, 0.4, 0.0001, 0.0005)
        {
            CountOfLayers = 0;
        }

        /// <summary>
        /// Initialise une nouvelle instance de la classe <see cref="NNetwork"/>.
        /// </summary>
        /// <param name="countOfLayers">
        /// The count of layers.
        /// </param>
        /// <param name="alpha">
        /// The alpha.
        /// </param>
        /// <param name="alphaMax">
        /// The alpha max.
        /// </param>
        /// <param name="eMin">
        /// The e min.
        /// </param>
        /// <param name="eeMin">
        /// The ee min.
        /// </param>
        public NNetwork(int countOfLayers, double alpha, double alphaMax, double eMin, double eeMin)
        {
            CountOfLayers = countOfLayers;
            Layers = new Layer[countOfLayers];
            Alpha = alpha;
            AlphaMax = alphaMax;
            Emin = eMin;
            EEmin = eeMin;
            log = LogManager.GetCurrentClassLogger();
        }

        /// <summary>
        /// The backpropogation training.
        /// </summary>
        /// <param name="batchContainer">
        /// The batch container.
        /// </param>
        /// <param name="cancellationToken">
        /// The cancellation token.
        /// </param>
        public void Train(BatchContainer batchContainer, CancellationToken cancellationToken)
        {
            var SumE = 1.0;
            var count = 0;
            //&& !Console.KeyAvailable
            Notifier.Notify(string.Format("BackPropagation started..."), true);
            while (SumE > EEmin && !cancellationToken.IsCancellationRequested)
            {
                SumE = 0.0;
                var cyclecount = 0;
                foreach (var batch in batchContainer.DataItems)
                {
                    foreach (var item in batch)
                    {
                        Layers[0].CalculateOutputs(item);

                        for (var i = 1; i < Layers.Length; i++)
                        {
                            var data = Layers[i - 1].GetOutputs();
                            Layers[i].CalculateOutputs(data, Layers[i]);
                        }
                        if (Layers[CountOfLayers - 1].ActivationFunction == NeuroHelper.Softmax)
                        {
                            var data = new double[Layers[CountOfLayers - 1].CountOfNeurons];
                            data[(int)batchContainer.Etalones[cyclecount]] = 1;
                            CalculateGammas(data);
                        }
                        else
                        {
                            CalculateGammas(item);
                        }
                        for (var i = CountOfLayers - 1; i > 0; i--)
                        {
                            Layers[i].ReCalculateWeights(Layers[i - 1], Alpha);
                        }
                        Layers[0].ReCalculateWeights(item, Alpha);
                        cyclecount++;
                    }
                }
                cyclecount = 0;
                foreach (var batch in batchContainer.DataItems)
                {
                    foreach (var item in batch)
                    {
                        Layers[0].CalculateOutputs(item);
                        for (var i = 1; i < Layers.Length; i++)
                        {
                            var data = Layers[i - 1].GetOutputs();
                            Layers[i].CalculateOutputs(data, Layers[i]);
                        }
                        if (Layers[CountOfLayers - 1].ActivationFunction == NeuroHelper.Softmax)
                        {
                            var data = new double[Layers[CountOfLayers - 1].CountOfNeurons];
                            data[(int)batchContainer.Etalones[cyclecount]] = 1;
                        }
                        else
                        {
                            var data = Layers[CountOfLayers - 2].GetOutputs();
                            Layers[CountOfLayers - 1].CalculateOutputs(data, Layers[CountOfLayers - 1]);
                        }
                        //ReCalculateAlpha();
                        if (Layers[CountOfLayers - 1].ActivationFunction == NeuroHelper.Softmax)
                        {
                            var data = new double[Layers[CountOfLayers - 1].CountOfNeurons];
                            data[(int)batchContainer.Etalones[cyclecount]] = 1;
                            SumE += CalculateError(data, item);
                        }
                        else
                        {
                            SumE += CalculateError(item, item);
                        }
                        cyclecount++;
                    }
                }
                //if (count%1 == 0)
                //{
                //    TraceInformation();
                //}

                //foreach (var neuron in Layers[CountOfLayers - 1].Neurons)
                //{
                //    Console.WriteLine(neuron.Y+ "; "+cyclecount);
                //}
                Notifier.Notify(string.Format("BackPropagation. Epoche:{0}; Error is:{1:2}", count, SumE),true);
                log.Trace("BackPropagation. Epoche:{0}; Error is:{1}", count, SumE);
                count++;
            }
        }

        /// <summary>
        /// The test on etalones.
        /// </summary>
        /// <param name="batchContainer">
        /// The batch container.
        /// </param>
        public void TestOnEtalones(BatchContainer batchContainer)
        {
            var cyclecount = 0;
            var sumE = 0.0;
            foreach (var batch in batchContainer.DataItems)
            {
                foreach (var item in batch)
                {
                    //Notifier.Notify(string.Format("Cyclecount is: {0};", cyclecount));
                    Layers[0].CalculateOutputs(item);
                    for (var i = 1; i < Layers.Length; i++)
                    {
                        var data = Layers[i - 1].GetOutputs();
                        Layers[i].CalculateOutputs(data, Layers[i]);
                    }
                    if (Layers[CountOfLayers - 1].ActivationFunction == NeuroHelper.Softmax)
                    {
                        var data = new double[Layers[CountOfLayers - 1].CountOfNeurons];
                        data[(int)batchContainer.Etalones[cyclecount]] = 1;
                    }
                    else
                    {
                        var data = Layers[CountOfLayers - 2].GetOutputs();
                        Layers[CountOfLayers - 1].CalculateOutputs(data, Layers[CountOfLayers - 1]);
                    }
                    //ReCalculateAlpha();
                    if (Layers[CountOfLayers - 1].ActivationFunction == NeuroHelper.Softmax)
                    {
                        var data = new double[Layers[CountOfLayers - 1].CountOfNeurons];
                        data[(int)batchContainer.Etalones[cyclecount]] = 1;
                        sumE += CalculateError(data, item);
                    }
                    else
                    {
                        sumE += CalculateError(item, item);
                    }
                    cyclecount++;
                    Notifier.Notify(string.Format("Testing on validation data... Processed {0} samples", cyclecount),true);
                }
            }
            var percent = 100 - sumE / ((double)cyclecount) * 100;

            Notifier.Notify(string.Format("Testing data on etalones. Didn't recognized {0} files; Percent of correct symbols is {1} %;", sumE, Math.Round(percent, 2)));
        }

        /// <summary>
        /// The calculate gammas.
        /// </summary>
        /// <param name="etalones">
        /// The etalones.
        /// </param>
        private void CalculateGammas(double[] etalones)
        {

            for (var i = 0; i < Layers[CountOfLayers - 1].CountOfNeurons; i++)
            {
                // var temp = Layers[CountOfLayers - 1].Neurons[i].Gamma;
                Layers[CountOfLayers - 1].Neurons[i].Gamma = Layers[CountOfLayers - 1].Neurons[i].Y - etalones[i];
                // Console.WriteLine(temp-Layers[CountOfLayers - 1].Neurons[i].Gamma);
            }
            if (CountOfLayers < 2) return;
            for (var i = CountOfLayers - 2; i >= 0; i--)
            {
                for (var j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    Layers[i].Neurons[j].Gamma = 0;
                    foreach (var n in Layers[i + 1].Neurons)
                    {
                        Layers[i].Neurons[j].Gamma +=
                            n.Gamma *
                            Layers[i + 1].DerivativeActivationFunction(n.Y)
                            * n.Weights[j];
                    }
                }
            }
        }

        /// <summary>
        /// The save into text file.
        /// </summary>
        /// <param name="path">
        /// The path.
        /// </param>
        public void SaveIntoTextFile(string path)
        {
            using (var sw = new StreamWriter(path))
            {
                var xmlDoc = new System.Xml.XmlDocument();
                var serializer = new XmlSerializer(this.GetType());
                using (var ms = new MemoryStream())
                {
                    serializer.Serialize(ms, this);
                    ms.Position = 0;
                    xmlDoc.Load(ms);
                    sw.Write(xmlDoc.InnerXml);
                }
            }
        }

        /// <summary>
        /// The calculate error.
        /// </summary>
        /// <param name="etalones">
        /// The etalones.
        /// </param>
        /// <param name="batchFirst">
        /// The batch first.
        /// </param>
        /// <returns>
        /// The <see cref="double"/>.
        /// </returns>
        private double CalculateError(double[] etalones, double[] batchFirst)
        {
            var result = 0.0;
            Layers[0].CalculateOutputs(batchFirst);
            // CalculateGammas(etalones);
            for (var i = 1; i < Layers.Length; i++)
            {
                var data = Layers[i - 1].GetOutputs();
                Layers[i].CalculateOutputs(data, Layers[i]);
                //CalculateGammas(data);
            }

            var index = Array.IndexOf(etalones, 1.0);
            var outputs = Layers[CountOfLayers - 1].Neurons.Select(x => x.Y).ToArray();
            var value = outputs.Max();
            var winIndex = Array.IndexOf(outputs, value);
            //  Notifier.Notify(string.Format("Max value:{0}. Win index:{1}; EtaloneIndex:{2}", value, winIndex, index));

            return winIndex == index ? 0 : 1;
        }

        /// <summary>
        /// The trace information.
        /// </summary>
        private void TraceInformation()
        {
            var outputs = Layers[CountOfLayers - 1].Neurons.Select(x => x.Y).ToArray();
            log.Trace("**********************************");
            for (var i = 0; i < outputs.Length; i++)
            {
                Console.WriteLine("output[{0}]={1}", i, outputs[i]);
            }
            log.Trace("**********************************");
            Console.WriteLine("Sum:{0}", outputs.Sum());
        }

    }
}
