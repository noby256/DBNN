using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace NeuralNetwork.Definition
{
    /// <summary>
    /// The batch container.
    /// </summary>
    public class BatchContainer
	{
        /// <summary>
        /// Initialize new instance of class <see cref="BatchContainer"/>.
        /// </summary>
        /// <param name="batchSize">
        /// The batch size.
        /// </param>
        public BatchContainer(int batchSize)
	    {
	        BatchSize = batchSize;
	        DataItems = new List<List<double[]>>();
	    }

        /// <summary>
        /// Gets or sets the data items.
        /// </summary>
        public List<List<double[]>> DataItems { get; set; }

        /// <summary>
        /// Gets or sets the etalones.
        /// </summary>
        public double[] Etalones { get; set; }

        /// <summary>
        /// Gets or sets the batch size.
        /// </summary>
        public int BatchSize { get; set; }

        /// <summary>
        /// The read data.
        /// </summary>
        /// <param name="filePath">
        /// The file path.
        /// </param>
        /// <exception cref="Exception">
        /// </exception>
        public void ReadData(string filePath)
		{
			//d:\Диплом\neuralNetworks\NeuralNetwork.Definition\Data0\
            DataItems = new List<List<double[]>>();
			var paths = Directory.GetFiles(filePath);

            if (paths.Length < BatchSize)
			{
				throw new Exception("Incompatible batch size");
			}

            var data = new List<double[]>();
			for (var i = 0; i < paths.Length; i++)
			{
			    try
			    {
                    data.Add(ReadBmp(paths[i]));
			    }
			    catch (Exception)
			    {
			        Console.WriteLine("unable to save file: {0}",paths[i]);
			    }
                if ((i + 1) % BatchSize == 0 || i == paths.Length - 1)
				{
                    Console.WriteLine("Readed path: {0}", paths[i]);
					DataItems.Add(data);
                    data = new List<double[]>();
				}
			}
		}

        /// <summary>
        /// The get etalone numbers.
        /// </summary>
        /// <param name="path">
        /// The path.
        /// </param>
        public void GetEtaloneNumbers(string path)
        {
            Etalones = GetNumbers(path).ToArray();
        }

        /// <summary>
        /// The get numbers.
        /// </summary>
        /// <param name="filePath">
        /// The file path.
        /// </param>
        /// <returns>
        /// The <see cref="IEnumerable"/>.
        /// </returns>
        private static IEnumerable<double> GetNumbers(string filePath)
	    {
            using (var sr = new StreamReader(filePath))
            {
                while (!sr.EndOfStream)
                {
                    var value = Convert.ToInt32(sr.ReadLine());
                    yield return value;
                }
            }
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
        private static double[] ReadBmp(string path)
        {
            var bmp = new Bitmap(path);
            var a = new double[bmp.Width * bmp.Height];
            var index = 0;
            for (var i = 0; i < bmp.Width; i++)
            {
                for (var j = 0; j < bmp.Height; j++)
                {
                    Color originalColor = bmp.GetPixel(i, j);
                    a[index] = originalColor.R / 255.0;
                    index++;
                }
            }
			return a;
		}

	}
}
