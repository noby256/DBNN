using System;
using System.Threading;
using NeuralNetwork.Definition;
namespace NeuralNetwork
{
    /// <summary>
    /// The program.
    /// </summary>
    class Program
    {
        /// <summary>
        /// The main.
        /// </summary>
        /// <param name="args">
        /// The args.
        /// </param>
        private static void Main(string[] args)
        {
			var batches = new BatchContainer(10);
            batches.ReadData(@"C:\Test\Data1000\");
            batches.GetEtaloneNumbers(@"C:\Test\aaa1000.txt");
	        /*
			var rbmC = new RbmConfig
	        {
				InputDataType = DataType.NonBinary,
				Algorithm = TrainingAlgorithm.Hinton, K = 1, Alpha = 0.01
	        };
	        var rbm = new Rbm(rbmC);
			rbm.CountHiddenNeurons = 150;
	        rbm.CountVisibleNeurons = 841;
			rbm.InitializeWeights();
			rbm.Train(batches,1000);
			*/
            NNetwork n = new NNetwork(4, new ConsoleNotifier());
            //n.Layers[0] = new Layer(100, NeuroHelper.LinearFunction, NeuroHelper.DerivativeLinearFunction);
            //n.Layers[1] = new Layer(100, NeuroHelper.SigmoidalFunction, NeuroHelper.DerivativeSigmoidalFunction);
            //n.Layers[2] = new Layer(100, NeuroHelper.SigmoidalFunction, NeuroHelper.DerivativeSigmoidalFunction);
            //n.Layers[3] = new Layer(100, NeuroHelper.SigmoidalFunction, NeuroHelper.DerivativeSigmoidalFunction);
            //n.Layers[4] = new Layer(841, NeuroHelper.SigmoidalFunction, NeuroHelper.DerivativeSigmoidalFunction);
            n.Layers[0] = new Layer(500, NeuroHelper.SigmoidalFunction, NeuroHelper.DerivativeSigmoidalFunction,0.01, new ConsoleNotifier());
			n.Layers[1] = new Layer(1000, NeuroHelper.SigmoidalFunction, NeuroHelper.DerivativeSigmoidalFunction, 0.01, new ConsoleNotifier());
			n.Layers[2] = new Layer(2000, NeuroHelper.SigmoidalFunction, NeuroHelper.DerivativeSigmoidalFunction, 0.01, new ConsoleNotifier());
			n.Layers[3] = new Layer(10, NeuroHelper.Softmax, NeuroHelper.DerivativeSoftmaxFunction, 0.01, new ConsoleNotifier());
            n.Alpha = 0.01;
          
            n.Layers[0].InitializeLayer(784, 1000);
            var config = new RbmConfig
            {
                Algorithm = TrainingAlgorithm.Hinton,
                CountOfEpoches = 2,
                Alpha = 0.01,
                K = 3,
                LoggingEnabled = false,
                InputDataType = DataType.NonBinary
            };
            n.Layers[0].TrainAsRbm(batches, config);
            var batch = n.Layers[0].GetBatchContainer(10);
			n.Layers[1].InitializeLayer(500, 1000);
            config = new RbmConfig
            {
                Algorithm = TrainingAlgorithm.Hinton,
                CountOfEpoches = 2,
                Alpha = 0.005,
                K = 3,
                LoggingEnabled = false,
                InputDataType = DataType.NonBinary
            };
            n.Layers[1].TrainAsRbm(batch, config);
            batch = n.Layers[1].GetBatchContainer(10);
            n.Layers[2].InitializeLayer(1000, 1000);
            config = new RbmConfig
            {
                Algorithm = TrainingAlgorithm.Hinton,
                CountOfEpoches = 2,
                Alpha = 0.001,
                K = 3,
                LoggingEnabled = false,
                InputDataType = DataType.NonBinary
            };
            n.Layers[2].TrainAsRbm(batch, config);
            //batch = n.Layers[2].GetBatchContainer(10);
            n.Layers[3].InitializeLayer(2000, 1000);
            //config = new RbmConfig
            //{
            //    Algorithm = TrainingAlgorithm.Golovko,
            //    CountOfEpoches = 100,
            //    Alpha = 0.002,
            //    K = 1,
            //    LoggingEnabled = false,
            //    InputDataType = DataType.NonBinary
            //};

       //     n.Layers[3].TrainAsRbm(batch, config);
            //batch = n.Layers[3].GetBatchContainer(10);

            n.Train(batches, new CancellationToken());
            Console.WriteLine("asdasd");
            batches.ReadData(@"C:\Test\DataTest\");
            batches.GetEtaloneNumbers(@"C:\Test\etalones.txt");
            n.TestOnEtalones(batches);
            //batch = n.Layers[4].GetBatchContainer(10);


            //Rbm rbm = new Rbm();
            //rbm.Alpha = 0.01;
            //rbm.CountHiddenNeurons = 150;
            //rbm.CountVisibleNeurons = 841;
            //rbm.InitializeWeights();
            //rbm.Momentum = 0.9;

            //rbm.Train(batches,int.MaxValue);
        }

        /// <summary>
        /// The console notifier.
        /// </summary>
        public class ConsoleNotifier : INotifier
	    {
            /// <summary>
            /// The notify.
            /// </summary>
            /// <param name="value">
            /// The value.
            /// </param>
            public void Notify(string value, bool replaceLine)
		    {
			    Console.WriteLine(value);
		    }
	    }
    }





}