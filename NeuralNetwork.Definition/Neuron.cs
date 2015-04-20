using System;

namespace NeuralNetwork.Definition
{
    /// <summary>
    /// The neuron.
    /// </summary>
    [Serializable]
	public class Neuron
	{
        /// <summary>
        /// Gets or sets the count of weights.
        /// </summary>
        public int CountOfWeights { get; set; }

        /// <summary>
        /// Gets or sets the yi.
        /// </summary>
        public double[] Yi { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public double[] Weights { get; set; }

        /// <summary>
        /// Gets or sets the t.
        /// </summary>
        public double T { get; set; }

        /// <summary>
        /// Gets or sets the y.
        /// </summary>
        public double Y { get; set; }

        /// <summary>
        /// Gets or sets the s.
        /// </summary>
        public double S { get; set; }

        /// <summary>
        /// Gets or sets the gamma.
        /// </summary>
        public double Gamma { get; set; }

        /// <summary>
        /// Initialise une nouvelle instance de la classe <see cref="Neuron"/>.
        /// </summary>
        public Neuron()
            : this(10, 10)
        {
            
        }

        /// <summary>
        /// Initialise une nouvelle instance de la classe <see cref="Neuron"/>.
        /// </summary>
        /// <param name="countOfWeights">
        /// The count of weights.
        /// </param>
        /// <param name="sizeLearn">
        /// The size learn.
        /// </param>
        public Neuron(int countOfWeights, int sizeLearn)
		{
			Yi = new double[sizeLearn];
			for (int i = 0; i < sizeLearn; i++)
			{
				Yi[i] = 0;
			}
		    CountOfWeights = countOfWeights;
			Weights = new double[countOfWeights];
		}

        /// <summary>
        /// The initialize weights.
        /// </summary>
        /// <param name="r">
        /// The r.
        /// </param>
        public void InitializeWeights(Random r)
		{
			for (int i = 0; i < Weights.Length; i++)
			{
				Weights[i] = r.NextDouble()-0.5;
			}
		}
    }
}
