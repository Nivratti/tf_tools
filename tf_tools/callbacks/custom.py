import time
import tensorflow as tf

class CustomLoggerCallback(tf.keras.callbacks.Callback):
    """
    Purpose: When running on TPU, observed significantly higher loss values when outputting loss for 
    each batch (using verbose=1 in model.fit). However, at the end of each epoch, the metrics values 
    appear correct. Implemented a callback to accurately log metrics values at the end of each epoch to 
    address this discrepancy.
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            print("\n")
            print("Metrics at the end of epoch {}: ".format(epoch + 1))
            for metric_name, metric_value in logs.items():
                print(f"{metric_name}: {metric_value}")
            print("")