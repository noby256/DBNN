using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Definition
{
    /// <summary>
    /// The rbm config.
    /// </summary>
    public class RbmConfig
    {
        /// <summary>
        /// Gets or sets the algorithm.
        /// </summary>
        public TrainingAlgorithm Algorithm { get; set; }

        /// <summary>
        /// Gets or sets the alpha.
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// Gets or sets the k.
        /// </summary>
        public int K { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether logging enabled.
        /// </summary>
        public bool LoggingEnabled { get; set; }

        /// <summary>
        /// Gets or sets the count of epoches.
        /// </summary>
        public int CountOfEpoches { get; set; }

        /// <summary>
        /// Gets or sets the input data type.
        /// </summary>
        public DataType InputDataType { get; set; }
    }

    /// <summary>
    /// The data type.
    /// </summary>
    public enum DataType
    {
        /// <summary>
        /// The binary.
        /// </summary>
        Binary,

        /// <summary>
        /// The non binary.
        /// </summary>
        NonBinary
    }
}
