import tensorflow as tf
from loguru import logger

def get_device_strategy(num_cores_to_use=None):
    """
    Determines and returns the appropriate distribution strategy for TensorFlow operations.
    This function checks for available hardware (TPU, GPU) and configures the suitable
    TensorFlow distribution strategy, prioritizing TPUs, then multiple GPUs using MirroredStrategy,
    a single GPU using OneDeviceStrategy, and defaults to the standard strategy (likely on CPU)
    if no GPUs are found.

    Args:
        num_cores_to_use (int, optional): The number of cores to use for distributed training
                                          on GPU or TPU. If None, all available cores are used.

    Returns:
        tuple: A tuple containing the TensorFlow distribution strategy and a string indicating
               the device type used ("TPU", "Multiple GPUs", "Single GPU", "CPU").

    Usage:
        strategy, device_type = get_device_strategy(num_cores_to_use=2)
        num_replicas = strategy.num_replicas_in_sync
        logger.info(f"Device type: {device_type}, REPLICAS: {num_replicas}")
    """
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        logger.info('Running on TPU ', resolver.master())
        device_type = "TPU"
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        if num_cores_to_use:
            strategy = tf.distribute.TPUStrategy(resolver, devices=resolver.get_job_devices()[:num_cores_to_use])
        else:
            strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError:
        gpus = tf.config.list_physical_devices('GPU')
        num_gpus = len(gpus)

        if num_gpus:
            if num_gpus > 1:
                logger.info(f"Multiple GPUs are available, using MirroredStrategy. Count: {num_gpus}")
                if num_cores_to_use and num_cores_to_use <= num_gpus:
                    devices = [f'/gpu:{i}' for i in range(num_cores_to_use)]
                    strategy = tf.distribute.MirroredStrategy(devices=devices)
                    logger.info(f"As num_cores_to_use is set to {num_cores_to_use}. using only {devices}")
                else:
                    strategy = tf.distribute.MirroredStrategy()
                device_type = "Multiple GPUs"
            else:
                logger.info("One GPU is available, using OneDeviceStrategy on GPU.")
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
                device_type = "Single GPU"
        else:
            logger.info("No GPUs found, using default strategy (likely on CPU).")
            strategy = tf.distribute.get_strategy()
            device_type = "CPU"

    return strategy, device_type