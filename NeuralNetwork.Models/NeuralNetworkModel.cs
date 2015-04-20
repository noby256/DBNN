using System;
using System.Xml.Serialization;

namespace NeuralNetwork.Models
{
    /// <summary>
    /// The neural network model.
    /// </summary>
    [Serializable]
    [XmlRoot("Network")]
    public class NeuralNetworkModel
    {
        /// <summary>
        /// Gets or sets the count of layers.
        /// </summary>
        [XmlElement("CountOfLayers")]
        public int CountOfLayers { get; set; }

        /// <summary>
        /// Gets or sets the layers.
        /// </summary>
        [XmlArray("Layers")]
        [XmlArrayItem("Layer", typeof(LayerModel))]
        public LayerModel[] Layers { get; set; }

        /// <summary>
        /// Gets or sets the alpha.
        /// </summary>
        [XmlElement("Alpha")]
        public double Alpha { get; set; }

        /// <summary>
        /// Gets or sets the training set count.
        /// </summary>
        [XmlElement("TrainingSetCount")]
        public int TrainingSetCount { get; set; }
    }
}
