using System;
using System.Xml.Serialization;
using NeuralNetwork.Definition;

namespace NeuralNetwork.Models
{
    /// <summary>
    /// The rbm config model.
    /// </summary>
    [Serializable]
    public class RbmConfigModel
    {
        /// <summary>
        /// Gets or sets the algorithm.
        /// </summary>
        [XmlElement("TrainingAlgorithm")]
        public TrainingAlgorithm Algorithm { get; set; }

        /// <summary>
        /// Gets or sets the rbm alpha.
        /// </summary>
        [XmlElement("Alpha")]
        public double RbmAlpha { get; set; }

        /// <summary>
        /// Gets or sets the k parameter.
        /// </summary>
        [XmlElement("KParameter")]
        public int KParameter { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether logging enabled.
        /// </summary>
        [XmlElement("LoggingEnabled")]
        public bool LoggingEnabled { get; set; }

        /// <summary>
        /// Gets or sets the input data type.
        /// </summary>
        [XmlElement("InputDataType")]
        public DataType InputDataType { get; set; }

        /// <summary>
        /// Gets or sets the count of epoches.
        /// </summary>
        [XmlElement("CountOfEpoches")]
		public int CountOfEpoches { get; set; }
    }
}
