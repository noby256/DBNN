using System;
using System.IO;
using System.Threading;
using System.Windows;
using System.Windows.Forms;
using System.Xml.Serialization;
using NeuralNetwork.Definition;
using NeuralNetwork.Models;
using NeuralNetwork.UI.Notifiers;
using NeuralNetwork.UI.ViewModels;
using MessageBox = System.Windows.MessageBox;
using SaveFileDialog = Microsoft.Win32.SaveFileDialog;
using System.Reflection;

namespace NeuralNetwork.UI
{
    using System.Text;
    using System.Threading.Tasks;

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        /// <summary>
        /// The view model.
        /// </summary>
        private MainViewModel viewModel;

        /// <summary>
        /// The neural network.
        /// </summary>
        private NeuralNetworkModel neuralNetwork;

        /// <summary>
        /// The Notifier.
        /// </summary>
        private INotifier _notifier;

        /// <summary>
        /// The _cancellation token.
        /// </summary>
        private CancellationTokenSource _cancellationToken;

        private BatchContainer batchContainer;

        private Thread trainTheread;

        /// <summary>
        /// Constructor <see cref="MainWindow"/>.
        /// </summary>
        public MainWindow()
        {
            InitializeComponent();
            DirectoryInfo dir = Directory.GetParent(
                Directory.GetParent(
                Directory.GetParent(Directory.GetCurrentDirectory()).ToString()).ToString());
            NetworkConfigFilePath.Text = dir.ToString() + @"\Data2\NetworkConfig.xml";
            BatchesDirPath.Text = dir.ToString() + @"\Data2\Data2_batches";
            EtalonesFilePath.Text = dir.ToString() + @"\Data2\Data2_etalons.txt";
            
            EtalonesValidatePath.Text = dir.ToString() + @"\Data2\Data2_etalons_validation.txt";
            ValidationDataPath.Text = dir.ToString() + @"\Data2\Data2_batches";

            _cancellationToken = new CancellationTokenSource();
            _notifier = new TextBlockNotifier(LoggerTextBox);
            viewModel = new MainViewModel();
            DataContext = viewModel;
        }

        /// <summary>
        /// The train nn click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void TrainNnClick(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(BatchesDirPath.Text))
            {
                MessageBox.Show("Please set path to data");
                return;
            }
            if (string.IsNullOrEmpty(EtalonesFilePath.Text))
            {
                MessageBox.Show("Please set path to etalones");
                return;
            }
            if (string.IsNullOrEmpty(NetworkConfigFilePath.Text))
            {
                MessageBox.Show("Please set path to network config");
                return;
            }
            var batchSize = int.Parse(BatchSize.Text);
            batchContainer = new BatchContainer(batchSize);
            batchContainer.ReadData(BatchesDirPath.Text);
            batchContainer.GetEtaloneNumbers(EtalonesFilePath.Text);

            try
            {
                var serializer = new XmlSerializer(typeof(NeuralNetworkModel));
                var reader = new StreamReader(NetworkConfigFilePath.Text);
                neuralNetwork = (NeuralNetworkModel)serializer.Deserialize(reader);
                //neuralNetwork.validate();
                reader.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Incorrect config file or data path to batch. Error is: {0}", ex.InnerException.ToString());
                return;
            }
            trainTheread = new Thread(TrainTheread);
            trainTheread.Start();
            //Task.Factory.StartNew(() =>
            //{
            //    DisableControls();
            //    IndicatorStackPanel.Dispatcher.Invoke((Action)(() => IndicatorStackPanel.Visibility = Visibility.Visible));
            //    viewModel.CreateTrainNetwork(neuralNetwork, batches, _notifier, _cancellationToken.Token);
            //    MessageBox.Show("Training is finished");
            //    IndicatorStackPanel.Dispatcher.Invoke((Action)(() => IndicatorStackPanel.Visibility = Visibility.Hidden));
            //    EnableControls();
            //},
            //    TaskCreationOptions.LongRunning);
        }

        private void TrainTheread()
        {
            Dispatcher.Invoke((Action)(() =>  LoggerTextBox.Text = ""));
            DisableControls();
            IndicatorStackPanel.Dispatcher.Invoke((Action)(() => IndicatorStackPanel.Visibility = Visibility.Visible));
            viewModel.CreateTrainNetwork(neuralNetwork, batchContainer, _notifier, _cancellationToken.Token);
            //MessageBox.Show("Training is finished");
            viewModel.NeuralNetwork.SaveIntoTextFile("nn.txt");
            IndicatorStackPanel.Dispatcher.Invoke((Action)(() => IndicatorStackPanel.Visibility = Visibility.Hidden));
            EnableControls();
        }

        /// <summary>
        /// The save nn click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void SaveNnClick(object sender, RoutedEventArgs e)
        {
            if (viewModel.NeuralNetwork == null)
            {
                MessageBox.Show("Please create neural network (import settings via config file)");
            }

            var fileDialog = new SaveFileDialog
                                 {
                                     Title = "save Text File",
                                     Filter = "TXT files|*.xml"
                                 };
            fileDialog.ShowDialog();
            if (fileDialog.FileName != string.Empty)
            {
                try
                {
                    viewModel.NeuralNetwork.SaveIntoTextFile(fileDialog.FileName);
                    MessageBox.Show("Network has successfully saved!");
                }
                catch (Exception ex)
                {
                    MessageBox.Show(string.Format("Failed to save! Error is: {0}", ex.Message));
                    return;
                }
            }
        }

        /// <summary>
        /// The open data click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void OpenDataClick(object sender, RoutedEventArgs e)
        {
            var fileDialog = new FolderBrowserDialog();
            if (fileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                var filePath = fileDialog.SelectedPath;
                if (filePath != string.Empty)
                {
                    BatchesDirPath.Text = filePath;
                }
            }
        }

        /// <summary>
        /// The open config click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void OpenConfigClick(object sender, RoutedEventArgs e)
        {
            var fileDialog = new OpenFileDialog
                                 {
                                     Title = "Open Text File",
                                     Filter = "TXT files|*.xml"
                                 };
            if (fileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                NetworkConfigFilePath.Text = fileDialog.FileName;
            }
        }

        /// <summary>
        /// The open etalone click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void OpenEtaloneClick(object sender, RoutedEventArgs e)
        {
            var fileDialog = new OpenFileDialog
                                 {
                                     Title = "Open Text File",
                                     Filter = "TXT files|*.txt"
                                 };
            if (fileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                EtalonesFilePath.Text = fileDialog.FileName;
            }
        }

        /// <summary>
        /// The open etalone validation click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void OpenEtaloneValidationClick(object sender, RoutedEventArgs e)
        {
            var fileDialog = new OpenFileDialog
                                 {
                                     Title = "Open Text File",
                                     Filter = "TXT files|*.txt"
                                 };
            if (fileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                EtalonesValidatePath.Text = fileDialog.FileName;
            }
        }

        /// <summary>
        /// The open data validate click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void OpenDataValidateClick(object sender, RoutedEventArgs e)
        {
            var fileDialog = new FolderBrowserDialog();
            if (fileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                var filePath = fileDialog.SelectedPath;
                if (filePath != string.Empty)
                {
                    ValidationDataPath.Text = filePath;
                }
            }
        }

        /// <summary>
        /// The validate data click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void ValidateDataClick(object sender, RoutedEventArgs e)
        {
            if (viewModel.NeuralNetwork == null)
            {
                MessageBox.Show("Please set path to saved neural network or train it according to the selected config file");
                return;
            }
            if (string.IsNullOrEmpty(ValidationDataPath.Text))
            {
                MessageBox.Show("Please set path to data");
                return;
            }
            if (string.IsNullOrEmpty(EtalonesValidatePath.Text))
            {
                MessageBox.Show("Please set path to etalones");
                return;
            }
            IndicatorStackPanel.Dispatcher.Invoke((Action)(() => IndicatorStackPanel.Visibility = Visibility.Visible));
            //DisableControls();

            var batchSize = int.Parse(BatchSize.Text);
            var batches = new BatchContainer(batchSize);
            batches.ReadData(ValidationDataPath.Text);
            batches.GetEtaloneNumbers(EtalonesValidatePath.Text);
            Task.Factory.StartNew(() =>
            {
                DisableControls();
                viewModel.NeuralNetwork.TestOnEtalones(batches);
                MessageBox.Show("Validation is finished");
                IndicatorStackPanel.Dispatcher.Invoke((Action)(() => IndicatorStackPanel.Visibility = Visibility.Hidden));
                EnableControls();
            });
        }

        /// <summary>
        /// The stop training button click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void StopTrainingButtonClick(object sender, RoutedEventArgs e)
        {
            _cancellationToken.Cancel();
            trainTheread.Abort();
            IndicatorStackPanel.Dispatcher.Invoke((Action)(() => IndicatorStackPanel.Visibility = Visibility.Hidden));
            EnableControls();
        }

        /// <summary>
        /// The load saved nn button click.
        /// </summary>
        /// <param name="sender">
        /// The sender.
        /// </param>
        /// <param name="e">
        /// The e.
        /// </param>
        private void LoadSavedNnButtonClick(object sender, RoutedEventArgs e)
        {
            var fileDialog = new OpenFileDialog
                                 {
                                     Title = "Open Text File",
                                     Filter = "TXT files|*.xml"
                                 };
            if (fileDialog.ShowDialog() != System.Windows.Forms.DialogResult.OK)
            {
                return;
            }
            using (TextReader textReader = new StreamReader(fileDialog.FileName))
            {
                var deserializer = new XmlSerializer(typeof(NNetwork));
                try
                {
                    this.viewModel.NeuralNetwork = (NNetwork)deserializer.Deserialize(textReader);
                }
                catch (Exception ex)
                {
                    MessageBox.Show(string.Format("Error during parsing the xml file. Error is {0}", ex.Message));
                    return;
                }

                if (this.viewModel.NeuralNetwork == null || this.viewModel.NeuralNetwork.Layers == null)
                {
                    MessageBox.Show("this.viewModel.NeuralNetwork == null || this.viewModel.NeuralNetwork.Layers == null");
                    return;
                }
                MessageBox.Show("Successfully loaded data");
                foreach (var layer in this.viewModel.NeuralNetwork.Layers)
                {
                    layer.ActivationFunction =
                        NeuroHelper.SetActivationFunction(
                            layer.ActivationFunctionName);
                    layer.DerivativeActivationFunction =
                        NeuroHelper.SetDerivativeActivationFunction(
                            layer.DerivativeActivationFunctionName);
                }
                viewModel.NeuralNetwork.Notifier = new TextBlockNotifier(LoggerTextBox);
            }
        }
        
        private void DisableControls()
        {
            OpenConfigButton.Dispatcher.Invoke((Action)(() => OpenConfigButton.IsEnabled = false));
            SaveNnButton.Dispatcher.Invoke((Action)(() => SaveNnButton.IsEnabled = false));
            OpenDataButton.Dispatcher.Invoke((Action)(() => OpenDataButton.IsEnabled = false));
            OpenConfigButton.Dispatcher.Invoke((Action)(() => OpenEtalonesButton.IsEnabled = false));
            ValidateNnDataButton.Dispatcher.Invoke((Action)(() => ValidateNnDataButton.IsEnabled = false));
            OpenDataButton.Dispatcher.Invoke((Action)(() => OpenValidationDataButton.IsEnabled = false));
            StackPanel.Dispatcher.Invoke((Action)(() => StackPanel.IsEnabled = false));
            TrainNnButton.Dispatcher.Invoke((Action)(() => TrainNnButton.IsEnabled = false));
            OpenEtaloneValidationData.Dispatcher.Invoke((Action)(() => OpenEtaloneValidationData.IsEnabled = false));
            LoadSavedNnButton.Dispatcher.Invoke((Action)(() => LoadSavedNnButton.IsEnabled = false));
            Dispatcher.Invoke((Action)(() => StopTrainingButton.IsEnabled = true));
        }

        private void EnableControls()
        {
            TrainNnButton.Dispatcher.Invoke((Action)(() => TrainNnButton.IsEnabled = true));
            OpenEtaloneValidationData.Dispatcher.Invoke((Action)(() => OpenEtaloneValidationData.IsEnabled = true));
            OpenConfigButton.Dispatcher.Invoke((Action)(() => OpenConfigButton.IsEnabled = true));
            SaveNnButton.Dispatcher.Invoke((Action)(() => SaveNnButton.IsEnabled = true));
            OpenDataButton.Dispatcher.Invoke((Action)(() => OpenDataButton.IsEnabled = true));
            StackPanel.Dispatcher.Invoke((Action)(() => StackPanel.IsEnabled = true));
            OpenConfigButton.Dispatcher.Invoke((Action)(() => OpenEtalonesButton.IsEnabled = true));
            ValidateNnDataButton.Dispatcher.Invoke((Action)(() => ValidateNnDataButton.IsEnabled = true));
            OpenDataButton.Dispatcher.Invoke((Action)(() => OpenValidationDataButton.IsEnabled = true));
            LoadSavedNnButton.Dispatcher.Invoke((Action)(() => LoadSavedNnButton.IsEnabled = true));
            Dispatcher.Invoke((Action)(() => StopTrainingButton.IsEnabled = false));
        }
    }
}
