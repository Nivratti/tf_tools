import tensorflow as tf
import psutil
import gc
from tf_keras.backend import clear_session
from tf_keras.callbacks import Callback

class MemoryCallback(Callback):
    """Callback to release memory and monitor memory usage at the end of each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch to release memory and log memory usage.

        Args:
            epoch (int): Index of the epoch.
            logs (dict): Metric results for this epoch.
        """
        # Measure memory usage before clearing session and garbage collection
        memory_info_before = psutil.virtual_memory()
        used_memory_before = memory_info_before.used / (1024 ** 3)

        # Clear TensorFlow session and run garbage collection to release memory
        clear_session()
        gc.collect()

        # Measure memory usage after clearing session and garbage collection
        memory_info_after = psutil.virtual_memory()
        used_memory_after = memory_info_after.used / (1024 ** 3)

        # Log memory usage before and after memory release
        print(f"Epoch {epoch+1}: Memory used before cleanup: {used_memory_before:.2f} GB")
        print(f"Epoch {epoch+1}: Memory used after cleanup: {used_memory_after:.2f} GB")
