using System.ComponentModel;

namespace NeuralNetwork.UI.ViewModels
{
    /// <summary>
    /// The base view model.
    /// </summary>
    public abstract class BaseViewModel : INotifyPropertyChanged
	{
        /// <summary>
        /// The property changed.
        /// </summary>
        public event PropertyChangedEventHandler PropertyChanged;

        /// <summary>
        /// The on property changed.
        /// </summary>
        /// <param name="propertyName">
        /// The property name.
        /// </param>
        protected void OnPropertyChanged(string propertyName)
		{
			var handlers = PropertyChanged;
			if (handlers != null)
			{
				handlers(this, new PropertyChangedEventArgs(propertyName));
			}
		}
	}
}
