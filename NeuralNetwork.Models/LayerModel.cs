using System;
using System.Xml.Serialization;

namespace NeuralNetwork.Models
{
    /// <summary>
    /// The layer model.
    /// </summary>
    [Serializable]
    public class LayerModel
    {
        /// <summary>
        /// Gets or sets the count of neurons.
        /// </summary>
        [XmlElement("NeuronsCount")]
        public int CountOfNeurons { get; set; }

        /// <summary>
        /// Gets or sets the weights count.
        /// </summary>
        [XmlElement("WeightsCount")]
        public int WeightsCount { get; set; }

        /// <summary>
        /// Gets or sets the activation function.
        /// </summary>
        [XmlElement("ActivationFunction")]
        public string ActivationFunction { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether is binary.
        /// </summary>
        [XmlElement("IsBinary")]
        public bool IsBinary { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether pre train.
        /// </summary>
        [XmlElement("Pretrain")]
        public bool PreTrain { get; set; }

        /// <summary>
        /// Gets or sets the rbm configuration.
        /// </summary>
        [XmlElement("RbmConfig")]
        public RbmConfigModel RbmConfiguration { get; set; }

        /// <summary>
        /// Gets or sets the alpha.
        /// </summary>
        [XmlElement("Alpha")]
        public double Alpha{ get; set; }
    }
}
