using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork.Definition
{
    /// <summary>
    /// The neuro helper.
    /// </summary>
    public static class NeuroHelper
    {
        /// <summary>
        /// The sigmoidal function.
        /// </summary>
        /// <param name="S">
        /// The s.
        /// </param>
        /// <param name="layer">
        /// The layer.
        /// </param>
        /// <returns>
        /// The <see cref="double"/>.
        /// </returns>
        public static double SigmoidalFunction(double S, Layer layer = null)
        {
            return 1 / (1 + Math.Exp(-S));
        }

        /// <summary>
        /// The tan h function.
        /// </summary>
        /// <param name="S">
        /// The s.
        /// </param>
        /// <param name="layer">
        /// The layer.
        /// </param>
        /// <returns>
        /// The <see cref="double"/>.
        /// </returns>
        public static double TanHFunction(double S, Layer layer = null)
        {
            return Math.Tanh(S);
        }

        /// <summary>
        /// The linear function.
        /// </summary>
        /// <param name="S">
        /// The s.
        /// </param>
        /// <param name="layer">
        /// The layer.
        /// </param>
        /// <returns>
        /// The <see cref="double"/>.
        /// </returns>
        public static double LinearFunction(double S, Layer layer = null)
        {
            return S;
        }

        /// <summary>
        /// The derivative sigmoidal function.
        /// </summary>
        /// <param name="S">
        /// The s.
        /// </param>
        /// <returns>
        /// The <see cref="double"/>.
        /// </returns>
        public static double DerivativeSigmoidalFunction(double S)
        {
            return S * (1 - S);
        }

        /// <summary>
        /// The derivative tan h function.
        /// </summary>
        /// <param name="S">
        /// The s.
        /// </param>
        /// <returns>
        /// The <see cref="double"/>.
        /// </returns>
        public static double DerivativeTanHFunction(double S)
        {
            return 1 - Math.Pow(S, 2);
        }

        /// <summary>
        /// The derivative linear function.
        /// </summary>
        /// <param name="S">
        /// The s.
        /// </param>
        /// <returns>
        /// The <see cref="double"/>.
        /// </returns>
        public static double DerivativeLinearFunction(double S)
        {
            return 1;
        }

        /// <summary>
        /// The softmax.
        /// </summary>
        /// <param name="s">
        /// The s.
        /// </param>
        /// <param name="layer">
        /// The layer.
        /// </param>
        /// <returns>
        /// The <see cref="double"/>.
        /// </returns>
        public static double Softmax(double s, Layer layer)
        {
            return Math.Exp(s) / (layer.Neurons.Select(x => Math.Exp(x.S)).Sum());
        }

        /// <summary>
        /// The derivative softmax function.
        /// </summary>
        /// <param name="s">
        /// The s.
        /// </param>
        /// <returns>
        /// The <see cref="double"/>.
        /// </returns>
        public static double DerivativeSoftmaxFunction(double s)
        {
            return s * (1 - s);
        }

        /// <summary>
        /// The get inputs from the textfile.
        /// </summary>
        /// <param name="path">
        /// The path.
        /// </param>
        /// <returns>
        /// The <see cref="IEnumerable"/>.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// </exception>
        public static IEnumerable<double[]> GetInputsFromTheTextfile(string path)
        {
            if (String.IsNullOrEmpty(path))
            {
                throw new ArgumentException("Invalid Path");
            }

            var streamReader = new StreamReader(path);
            string str;
            while (!streamReader.EndOfStream)
            {
                str = streamReader.ReadLine();
                var inputs = str.Split(' ');
                var res = inputs.Select(Convert.ToDouble);
                yield return res.ToArray();
            }
        }

        /// <summary>
        /// The set activation function.
        /// </summary>
        /// <param name="name">
        /// The name.
        /// </param>
        /// <returns>
        /// The <see cref="Func"/>.
        /// </returns>
        public static Func<double, Layer, double> SetActivationFunction(string name)
        {
            switch (name)
            {
                case "SigmoidalFunction":
                    {
                        return SigmoidalFunction;
                    }
                case "TanHFunction":
                    {
                        return TanHFunction;
                    }
                case "LinearFunction":
                    {
                        return LinearFunction;
                    }
                case "Softmax":
                    {
                        return Softmax;
                    }
                default:
                    {
                        return LinearFunction;
                    }
            }
        }

        /// <summary>
        /// The set derivative activation function.
        /// </summary>
        /// <param name="name">
        /// The name.
        /// </param>
        /// <returns>
        /// The <see cref="Func"/>.
        /// </returns>
        public static Func<double, double> SetDerivativeActivationFunction(string name)
        {
            switch (name)
            {
                case "SigmoidalFunction":
                    {
                        return DerivativeSigmoidalFunction;
                    }
                case "TanHFunction":
                    {
                        return DerivativeTanHFunction;
                    }
                case "LinearFunction":
                    {
                        return DerivativeLinearFunction;
                    }
                case "Softmax":
                    {
                        return DerivativeSoftmaxFunction;
                    }
                default:
                    {
                        return DerivativeLinearFunction;
                    }
            }
        }
    }
}
