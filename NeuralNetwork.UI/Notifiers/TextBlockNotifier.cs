using System;
using System.Threading;
using System.Windows.Controls;
using NeuralNetwork.Definition;

namespace NeuralNetwork.UI.Notifiers
{
    /// <summary>
    /// The text block notifier.
    /// </summary>
    public class TextBlockNotifier : INotifier
    {
        /// <summary>
        /// The _block.
        /// </summary>
        private readonly TextBlock _block;

        /// <summary>
        /// Initialise une nouvelle instance de la classe <see cref="TextBlockNotifier"/>.
        /// </summary>
        /// <param name="block">
        /// The block.
        /// </param>
        public TextBlockNotifier(TextBlock block)
        {
            this._block = block;
        }

        /// <summary>
        /// The notify.
        /// </summary>
        /// <param name="value">
        /// The value.
        /// </param>
        public void Notify(string value, bool replaceLine = false)
        {
            _block.Dispatcher.Invoke(
            () =>
            {
                if (_block.Text.Length > 1000)
                {
                    _block.Text = _block.Text.Remove(0, 600);
                }
                if (replaceLine)
                {
                    _block.Text = _block.Text.Remove(_block.Text.LastIndexOf("\n"));
                }
                _block.Text += "\n" + value;
            });
        }
    }
}
