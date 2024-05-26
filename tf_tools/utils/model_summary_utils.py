import tensorflow as tf
import io
from contextlib import redirect_stdout

def save_model_summary_to_file(model, file_path):
    """
    Saves the summary of a TensorFlow Keras model to a text file.

    Args:
        model (tf.keras.Model): The model whose summary is to be saved.
        file_path (str): The path to the file where the summary will be saved.

    Usage Example:
        model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=True)
        save_model_summary_to_file(model, 'model_summary.txt')
    """
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()

def show_partial_model_summary(model, max_layers=10):
    """
    Prints a partial summary of a TensorFlow Keras model, showing up to a specified number of layers.

    Args:
        model (tf.keras.Model): The model whose summary is to be printed.
        max_layers (int): The maximum number of layers to print in the summary.

    Usage Example:
        model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=True)
        partial_model_summary(model, max_layers=10)
    """
    # Calculate the total number of parameters
    total_params = model.count_params()
    # Each parameter is a 32-bit float, which is 4 bytes
    bytes_per_param = 4
    # Convert the total parameters to megabytes
    total_params_in_mb = int((total_params * bytes_per_param) / (1024 ** 2))
    # Print the total number of parameters
    print(f"Total parameters: {total_params_in_mb} MB \n")

    # --------
    layers = model.layers
    total_layers = len(layers)
    print(f"Model Summary (First {max_layers} layers out of {total_layers}):")
    for i, layer in enumerate(layers[:max_layers]):
        print(f"Layer {i+1}: {layer.name} ({layer.__class__.__name__})")
        output_shape = layer.output_shape
        param_count = layer.count_params()
        print(f"  Output Shape: {output_shape}")
        print(f"  Number of Parameters: {param_count}")
    if total_layers > max_layers:
        print(f"... (remaining {total_layers - max_layers} layers not shown)")

# # Example usage for saving model summary to a file
# model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=True)
# save_model_summary_to_file(model, 'model_summary.txt')

# # Example usage for displaying partial model summary
# show_partial_model_summary(model, max_layers=10)
