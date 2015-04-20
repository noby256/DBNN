using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NLog;

namespace NeuralNetwork.Definition
{

	public class Rbm
	{
		private bool loggingEnabled = Convert.ToBoolean(ConfigurationManager.AppSettings["loggingEnabled"]);

	    /// <summary>
	    /// The _log.
	    /// </summary>
	    private static Logger _log;

	    /// <summary>
	    /// The log per epoche.
	    /// </summary>
	    private const int LogPerEpoche = 20;

	    /// <summary>
	    /// Gets or sets the w.
	    /// </summary>
	    public double[,] W { get; set; }

	    /// <summary>
	    /// Gets or sets the tin.
	    /// </summary>
	    public double[] Tin { get; set; }

	    /// <summary>
	    /// Gets or sets the tout.
	    /// </summary>
	    public double[] Tout { get; set; }

	    /// <summary>
	    /// Gets or sets the count visible neurons.
	    /// </summary>
	    public int CountVisibleNeurons { get; set; }

	    /// <summary>
	    /// Gets or sets the count hidden neurons.
	    /// </summary>
	    public int CountHiddenNeurons { get; set; }

	    /// <summary>
	    /// Gets or sets the x.
	    /// </summary>
	    public double[] X { get; set; }

	    /// <summary>
	    /// Gets or sets the x rec.
	    /// </summary>
	    public double[] XRec { get; set; }

	    /// <summary>
	    /// Gets or sets the y.
	    /// </summary>
	    public double[] Y { get; set; }

	    /// <summary>
	    /// Gets or sets the alpha.
	    /// </summary>
	    public double Alpha { get; set; }

	    /// <summary>
	    /// Gets or sets the sum e.
	    /// </summary>
	    public double SumE { get; set; }

	    /// <summary>
	    /// Gets or sets the e.
	    /// </summary>
	    public double E { get; set; }

	    /// <summary>
	    /// The cd k.
	    /// </summary>
	    public int CdK = 1;

	    /// <summary>
	    /// The batch size.
	    /// </summary>
	    public int BatchSize = 10;

	    /// <summary>
	    /// The _lock.
	    /// </summary>
	    private readonly object _lock = new object();

	    /// <summary>
	    /// Gets or sets the momentum.
	    /// </summary>
	    public double Momentum { get; set; }

	    /// <summary>
	    /// Gets or sets the batches container.
	    /// </summary>
	    public BatchContainer BatchesContainer { get; set; }

	    /// <summary>
	    /// Gets or sets the algorithm.
	    /// </summary>
	    public TrainingAlgorithm Algorithm { get; set; }

	    /// <summary>
	    /// Gets or sets the input data type.
	    /// </summary>
	    public DataType InputDataType { get; set; }

	    /// <summary>
	    /// Initialise une nouvelle instance de la classe <see cref="Rbm"/>.
	    /// </summary>
	    /// <param name="rbmConfig">
	    /// The rbm config.
	    /// </param>
	    public Rbm(RbmConfig rbmConfig)
		{
			CdK = rbmConfig.K;
			Alpha = rbmConfig.Alpha;
			Algorithm = rbmConfig.Algorithm;
			InputDataType = rbmConfig.InputDataType;
		}

	    /// <summary>
	    /// The initialize weights.
	    /// </summary>
	    public void InitializeWeights()
		{
			W = new double[CountVisibleNeurons, CountHiddenNeurons];
			Y = new double[CountHiddenNeurons];
			X = new double[CountVisibleNeurons];
			XRec = new double[CountVisibleNeurons];
			Tin = new double[CountHiddenNeurons];
			Tout = new double[CountVisibleNeurons];
			var r = new Random();

			//N(0, 0.001),
			for (var i = 0; i < CountVisibleNeurons; i++)
				for (var j = 0; j < CountHiddenNeurons; j++)
				{
					W[i, j] = r.NextDouble() - 0.5;
				}

			for (var i = 0; i < CountHiddenNeurons; i++)
                Tin[i] = r.NextDouble() - 0.5;

			for (var i = 0; i < CountVisibleNeurons; i++)
                Tout[i] = r.NextDouble() - 0.5;
		}

	    /// <summary>
	    /// The train.
	    /// </summary>
	    /// <param name="batches">
	    /// The batches.
	    /// </param>
	    /// <param name="countOfEpoches">
	    /// The count of epoches.
	    /// </param>
	    /// <param name="notifier">
	    /// The notifier.
	    /// </param>
	    public void Train(BatchContainer batches, int countOfEpoches, INotifier notifier)
		{
			_log = LogManager.GetCurrentClassLogger();


			BatchesContainer = batches;
			var r = new Random();
			SumE = double.MaxValue;
			_log.Trace("Contrastive divergence starts...");
			notifier.Notify("Contrastive divergence starts...");
			var count = 0;

			while (SumE >= 0.1)
			{
              //  Save(count);
				switch (Algorithm)
				{
					case TrainingAlgorithm.Golovko:
						{
							Parallel.ForEach(batches.DataItems, TrainBatchGolovko);
							break;
						}
					case TrainingAlgorithm.Hinton:
						{
							Parallel.ForEach(batches.DataItems, TrainBatch);
							break;
						}
				}

				#region Расстояние Хэмминга (вычисление ошибки)

				SumE = 0;
				Parallel.ForEach(batches.DataItems, CalculateError);
				SumE = SumE / BatchesContainer.DataItems.Count;
                notifier.Notify(string.Format("Epoche:{0}; Error is:{1}", count, SumE), true);
                _log.Trace("Epoche:{0}; Error is:{1}", count, SumE);
				//Console.WriteLine("SumE={0};", SumE);
				count++;
               
                if (count % 100 == 0)
                {
                    //SaveRecognizedInputs();
                   // Save(count);
                }
				if (count == countOfEpoches)
				{
					return;
				}
				#endregion
			}
		}

	    /// <summary>
	    /// The train batch.
	    /// </summary>
	    /// <param name="batch">
	    /// The batch.
	    /// </param>
	    public void TrainBatch(List<double[]> batch)
		{
			Random r = new Random();
			var Y = new double[CountHiddenNeurons];
			var X = new double[CountVisibleNeurons];
			var XRec = new double[CountVisibleNeurons];
			var TIn = new double[CountHiddenNeurons];
			var TOut = new double[CountVisibleNeurons];


			//nabla
			double[,] nablaWeights = new double[CountVisibleNeurons, CountHiddenNeurons];
			double[] nablaTIn = new double[CountHiddenNeurons];
			double[] nablaTOut = new double[CountVisibleNeurons];

			#region Iterate through batch
			foreach (var input in batch)
			{
				//инициализируем видимые нейроны
				for (var i = 0; i < input.Length; i++)
				{
					X[i] = input[i];
				}

				#region Gibbs sampling

				for (int k = 0; k <= CdK; k++)
				{
					for (int i = 0; i < CountHiddenNeurons; i++)
					{
						Y[i] = 0;
					}
					//Calculate Hidden layer
					for (var i = 0; i < CountHiddenNeurons; i++)
					{
						for (var j = 0; j < CountVisibleNeurons; j++)
						{
							Y[i] += W[j, i] * X[j];
						}
						Y[i] += TIn[i];

						Y[i] = NeuroHelper.SigmoidalFunction(Y[i]);
					}


					#region accumulate negative phase

					//Считаем отрицательную часть градиента, если у нас последняя итерация алгоритма CD-k
					if (k == CdK)
					{
						for (var i = 0; i < CountVisibleNeurons; i++)
						{
							for (var j = 0; j < CountHiddenNeurons; j++)
							{
								nablaWeights[i, j] -= X[i] * Y[j];
							}
						}

						//циклы для набл смещений
						for (var i = 0; i < CountHiddenNeurons; i++)
						{
							nablaTIn[i] -= Y[i];
						}

						for (var i = 0; i < CountVisibleNeurons; i++)
						{
							nablaTOut[i] -= X[i];
						}
						break;
					}

					#endregion

					//семплирование скрытых нейронов
					for (int i = 0; i < CountHiddenNeurons; i++)
					{
						Y[i] = r.NextDouble() <= Y[i] ? 1d : 0d;
					}

					#region accumulate positive phase

					//Считаем положительную часть градиента, если это первая итерация
					if (k == 0)
					{
						for (int i = 0; i < CountVisibleNeurons; i++)
						{
							for (int j = 0; j < CountHiddenNeurons; j++)
							{
								nablaWeights[i, j] += X[i] * Y[j];
							}
						}

						//циклы для набл смещений
						for (var i = 0; i < CountHiddenNeurons; i++)
						{
							nablaTIn[i] += Y[i];
						}

						for (var i = 0; i < CountVisibleNeurons; i++)
						{
							nablaTOut[i] += X[i];
						}
					}

					#endregion

					//Восстанавливаем образ и получаем x'[1]
					for (var i = 0; i < CountVisibleNeurons; i++)
					{
						XRec[i] = 0;
					}

					for (var i = 0; i < CountVisibleNeurons; i++)
					{
						for (var j = 0; j < CountHiddenNeurons; j++)
						{
							XRec[i] += W[i, j] * Y[j];
						}
						XRec[i] += TOut[i];
						XRec[i] = NeuroHelper.SigmoidalFunction(XRec[i]);
					}


					for (int i = 0; i < CountVisibleNeurons; i++)
					{
						X[i] = XRec[i];
					}
				}
				#endregion
			}
			#endregion
			#region weights
			//Пересчитываем веса
			for (int i = 0; i < CountVisibleNeurons; i++)
			{
				for (int j = 0; j < CountHiddenNeurons; j++)
				{
					lock (_lock)
					{
						W[i, j] += Alpha * (nablaWeights[i, j] / BatchSize);
					}
				}
			}

			////Пересчитываем пороги
			for (var i = 0; i < CountHiddenNeurons; i++)
			{
				lock (_lock)
				{
					TIn[i] += Alpha * +nablaTIn[i] / BatchSize;
				}

			}

			for (var i = 0; i < CountVisibleNeurons; i++)
			{
				lock (_lock)
				{
					TOut[i] += Alpha * +nablaTOut[i] / BatchSize;
				}
			}

			//считаем ошибку
			E = 0;
			#endregion
		}

	    /// <summary>
	    /// The train batch golovko.
	    /// </summary>
	    /// <param name="batch">
	    /// The batch.
	    /// </param>
	    private void TrainBatchGolovko(IEnumerable<double[]> batch)
		{
			var r = new Random();
			var y = new double[CountHiddenNeurons];
			var x = new double[CountVisibleNeurons];
			var xRec = new double[CountVisibleNeurons];


			//nabla
			var nablaWeights = new double[CountVisibleNeurons, CountHiddenNeurons];
			var nablaTIn = new double[CountHiddenNeurons];
			var nablaTOut = new double[CountVisibleNeurons];
			var y0 = new double[CountHiddenNeurons];
			var yk = new double[CountHiddenNeurons];
			var x0 = new double[CountVisibleNeurons];
			var xk = new double[CountVisibleNeurons];

			#region Iterate through batch
			foreach (var input in batch)
			{
				//инициализируем видимые нейроны
				for (var i = 0; i < input.Length; i++)
				{
					x[i] = input[i];
				}
				for (var i = 0; i < CountHiddenNeurons; i++)
				{
					y[i] = 0;
				}
				#region Gibbs sampling

				for (var k = 0; k <= CdK; k++)
				{
					for (var i = 0; i < CountHiddenNeurons; i++)
					{
						y[i] = 0;
					}
					//Calculate Hidden layer
					for (var i = 0; i < CountHiddenNeurons; i++)
					{
						for (var j = 0; j < CountVisibleNeurons; j++)
						{
							y[i] += W[j, i] * x[j];
						}
						y[i] += Tin[i];

						y[i] = NeuroHelper.SigmoidalFunction(y[i]);
						// Console.WriteLine(Y[i]);
					}

					#region logging
					//if (count % LogPerEpoche == 0 && loggingEnabled)
					//{
					//    log.Debug("****************\nHidden propabilities UNSAMPLED:");
					//    for (var j = 0; j < CountHiddenNeurons; j++)
					//        if (j % 10 == 0) log.Debug("Y[{0}]={1}", j, Y[j]);

					//}
					#endregion

					#region accumulate negative phase

					//Считаем отрицательную часть градиента, если у нас последняя итерация алгоритма CD-k
					if (k == CdK)
					{
						for (var i = 0; i < CountVisibleNeurons; i++)
						{
							xk[i] = x[i];
						}
						for (var i = 0; i < CountHiddenNeurons; i++)
						{
							yk[i] = y[i];
						}

						for (var i = 0; i < CountVisibleNeurons; i++)
						{
							for (var j = 0; j < CountHiddenNeurons; j++)
							{
								nablaWeights[i, j] += (yk[j] - y0[j]) *
									yk[j] * (1 - yk[j]) * xk[i] +
									(xk[i] - x0[i]) *
									xk[i] * (1 - xk[i]) * y0[j];
							}
						}

						//циклы для набл смещений
						for (var j = 0; j < CountHiddenNeurons; j++)
						{
							nablaTIn[j] += (yk[j] - y0[j]) * yk[j] * (1 - yk[j]);
						}

						for (var i = 0; i < CountVisibleNeurons; i++)
						{
							nablaTOut[i] += (xk[i] - x0[i]) * xk[i] * (1 - xk[i]);
						}
						break;
					}

					#endregion

					//Считаем положительную часть градиента, если это первая итерация
					if (k == 0)
					{
						for (var j = 0; j < CountHiddenNeurons; j++)
						{
							y0[j] = y[j];
						}
						for (var j = 0; j < CountVisibleNeurons; j++)
						{
							x0[j] = x[j];
						}
					}

				#endregion

					//Восстанавливаем образ и получаем x'[1]
					for (var i = 0; i < CountVisibleNeurons; i++)
					{
						xRec[i] = 0;
					}

					for (var i = 0; i < CountVisibleNeurons; i++)
					{
						for (var j = 0; j < CountHiddenNeurons; j++)
						{
							xRec[i] += W[i, j] * y[j];
						}
						xRec[i] += Tout[i];
						xRec[i] = NeuroHelper.SigmoidalFunction(xRec[i]);
					}


					for (var i = 0; i < CountVisibleNeurons; i++)
					{
						x[i] = xRec[i];
					}

				}
			#endregion
			}
			#region weights
			//Пересчитываем веса
			for (var i = 0; i < CountVisibleNeurons; i++)
			{
				for (var j = 0; j < CountHiddenNeurons; j++)
				{
					lock (_lock)
					{
						W[i, j] -= Alpha * (nablaWeights[i, j] / BatchSize);
					}
				}
			}

			//Пересчитываем пороги
			for (var i = 0; i < CountHiddenNeurons; i++)
			{
				lock (_lock)
				{
					Tin[i] -= Alpha * +nablaTIn[i] / BatchSize;
				}

			}

			for (var i = 0; i < CountVisibleNeurons; i++)
			{
				lock (_lock)
				{
					Tout[i] -= Alpha * +nablaTOut[i] / BatchSize;
				}
			}

			//считаем ошибку
			E = 0;
			#endregion

		}

	    /// <summary>
	    /// The save.
	    /// </summary>
	    /// <param name="count">
	    /// The count.
	    /// </param>
	    private void Save(int count)
        {
            var visible = (int)Math.Round(Math.Sqrt(CountVisibleNeurons));
            var hidden = visible;

            var bmp = new Bitmap(visible * hidden, visible * hidden);
            for (var i = 0; i < CountHiddenNeurons; i++)
            {
                var data = new double[CountVisibleNeurons];
                for (var j = 0; j < CountVisibleNeurons; j++)
                {
                    data[j] = W[j, i];
                }
                var max = data.Max();
                var min = data.Min();
                for (var j = 0; j < CountVisibleNeurons; j++)
                {

                    var value = (W[j, i] + Math.Abs(min)) / (Math.Abs(min) + max);
                    var x = visible * (i / hidden) + j / visible;
                    var y = visible * (i % hidden) + j % visible;
                    bmp.SetPixel(x, y,
                        Color.FromArgb(
                        (int)Math.Round(value * 255),
                        (int)Math.Round(value * 255),
                        (int)Math.Round(value * 255)));
                }
            }
            bmp.Save(string.Format(@"PATTERN{0}.{1}х{2}.bmp", count,CountVisibleNeurons,CountHiddenNeurons));
        }

	    /// <summary>
	    /// The calculate error.
	    /// </summary>
	    /// <param name="batch">
	    /// The batch.
	    /// </param>
	    public void CalculateError(List<double[]> batch)
        {
            var r = new Random();
            var y = new double[CountHiddenNeurons];
            var x = new double[CountVisibleNeurons];
            var xRec = new double[CountVisibleNeurons];
            var TIn = new double[CountHiddenNeurons];
            var TOut = new double[CountVisibleNeurons];

            double E = 0;
            foreach (var input in batch)
            {
                for (var i = 0; i < input.Length; i++)
                {
                    x[i] = input[i];
                    xRec[i] = 0;
                }
                for (var j = 0; j < CountHiddenNeurons; j++)
                {
                    y[j] = 0;
                }
                for (var i = 0; i < CountHiddenNeurons; i++)
                {
                    for (var j = 0; j < CountVisibleNeurons; j++)
                    {
                        y[i] += W[j, i] * x[j];
                    }

                    y[i] += TIn[i];
                    y[i] = NeuroHelper.SigmoidalFunction(y[i]);

                    y[i] = r.NextDouble() <= y[i] ? 1d : 0d;
                }

                for (var i = 0; i < CountVisibleNeurons; i++)
                {
                    for (var j = 0; j < CountHiddenNeurons; j++)
                    {
                        xRec[i] += W[i, j] * y[j];
                    }
                    xRec[i] += TOut[i];
                    xRec[i] = NeuroHelper.SigmoidalFunction(xRec[i]);
                   if (InputDataType==DataType.Binary) xRec[i] = r.NextDouble() <= xRec[i] ? 1d : 0d;
                }

                // Console.ReadKey();
                for (var i = 0; i < CountVisibleNeurons; i++)
                {
                    lock (_lock)
                    {
                        E += Math.Abs(x[i] - xRec[i]);
                    }
                }

            }
            E = E / BatchSize;
            SumE += E;
        }

	    /// <summary>
	    /// The read bmp.
	    /// </summary>
	    /// <param name="path">
	    /// The path.
	    /// </param>
	    /// <returns>
	    /// The <see cref="double[]"/>.
	    /// </returns>
	    public double[] ReadBmp(string path)
		{
			var bmp = new Bitmap(path);

			var a = new double[bmp.Width * bmp.Height];
			var index = 0;
			for (var i = 0; i < bmp.Width; i++)
			{
				for (var j = 0; j < bmp.Height; j++)
				{
					a[index] = bmp.GetPixel(i, j).G == 255 ? 1 : 0;
					index++;
				}

			}

			return a;
		}

	    /// <summary>
	    /// The save recognized inputs.
	    /// </summary>
	    public void SaveRecognizedInputs()
		{
			var idx = 0;
			foreach (var input in BatchesContainer.DataItems.SelectMany(inputs => inputs))
			{
				for (var i = 0; i < input.Length; i++)
				{
					X[i] = input[i];
					XRec[i] = 0;
				}
				for (var j = 0; j < CountHiddenNeurons; j++)
				{
					Y[j] = 0;
				}

				var r = new Random();

				for (var i = 0; i < CountHiddenNeurons; i++)
				{
					for (var j = 0; j < CountVisibleNeurons; j++)
					{
						Y[i] += W[j, i] * X[j];
					}
					Y[i] += Tin[i];
					Y[i] = NeuroHelper.SigmoidalFunction(Y[i]);
					Y[i] = r.NextDouble() <= Y[i] ? 1d : 0d;
				}

				for (var i = 0; i < CountVisibleNeurons; i++)
				{
					for (var j = 0; j < CountHiddenNeurons; j++)
					{
						XRec[i] += W[i, j] * Y[j];
					}
					XRec[i] += Tout[i];
					XRec[i] = NeuroHelper.SigmoidalFunction(XRec[i]);
					XRec[i] = r.NextDouble() <= XRec[i] ? 1d : 0d;
				}

				var bmp = new Bitmap(29, 29);
				var index = 0;
				for (var i = 0; i < bmp.Width; i++)
				{
					for (var j = 0; j < bmp.Height; j++)
					{
						bmp.SetPixel(i, j, XRec[index] < 1 ? Color.Black : Color.White);
						index++;
					}
				}

				bmp.Save(string.Format(@"rec{0}.bmp", idx++));


				bmp = new Bitmap(29, 29);
				index = 0;
				for (var i = 0; i < bmp.Width; i++)
				{
					for (var j = 0; j < bmp.Height; j++)
					{
						bmp.SetPixel(i, j, X[index] < 1 ? Color.Black : Color.White);
						index++;
					}
				}

				bmp.Save(string.Format(@"etalone{0}.bmp", idx));
			}
		}

	    /// <summary>
	    /// The get etalones.
	    /// </summary>
	    /// <param name="batchSize">
	    /// The batch size.
	    /// </param>
	    /// <returns>
	    /// The <see cref="List"/>.
	    /// </returns>
	    public List<double[]> GetEtalones()
		{
			var r = new Random();
			var result = new List<double[]>();
			foreach (var input in BatchesContainer.DataItems.SelectMany(inputs => inputs))
			{
				for (var i = 0; i < input.Length; i++)
				{
					X[i] = input[i];
					XRec[i] = 0;
				}
				for (var j = 0; j < CountHiddenNeurons; j++)
				{
					Y[j] = 0;
				}

				for (var i = 0; i < CountHiddenNeurons; i++)
				{
					for (var j = 0; j < CountVisibleNeurons; j++)
					{
						Y[i] += W[j, i] * X[j];
					}
					Y[i] += Tin[i];
					Y[i] = NeuroHelper.SigmoidalFunction(Y[i]);
					if (InputDataType == DataType.Binary) Y[i] = r.NextDouble() <= Y[i] ? 1d : 0d;
				}
				var item = new double[CountHiddenNeurons];
				Array.Copy(Y, item, item.Length);
				result.Add(item);
			}
			return result;
		}
	}

    /// <summary>
    /// The training algorithm.
    /// </summary>
    public enum TrainingAlgorithm
	{
        /// <summary>
        /// The hinton.
        /// </summary>
        Hinton,

        /// <summary>
        /// The golovko.
        /// </summary>
        Golovko
	}
}
