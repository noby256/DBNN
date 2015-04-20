namespace NeuralNetwork.Definition
{
    /// <summary>
    /// The Notifier interface.
    /// </summary>
    public interface INotifier
	{
        /// <summary>
        /// The notify.
        /// </summary>
        /// <param name="value">
        /// The value.
        /// </param>
        void Notify(string value, bool replaceLine = false);
	}
}
