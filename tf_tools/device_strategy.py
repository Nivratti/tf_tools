import tensorflow as tf
from loguru import logger

def get_device_statergy():
    """Determines and returns the appropriate distribution strategy for TensorFlow operations.

    This function checks for available hardware (TPU, GPU) and returns the suitable
    TensorFlow distribution strategy. It prioritizes TPUs, then multiple GPUs using
    MirroredStrategy, a single GPU using OneDeviceStrategy, and defaults to the standard
    strategy (likely on CPU) if no GPUs are found.

    Returns:
        tf.distribute.Strategy: The TensorFlow distribution strategy based on available hardware.

    Usage:
    strategy, is_tpu = get_device_statergy()
    num_replicas = strategy.num_replicas_in_sync
    logger.info(f"REPLICAS: {num_replicas}")
    """
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        logger.info('Running on TPU ', resolver.master())
        is_tpu = True
    except ValueError:
        is_tpu = None

    if is_tpu:
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        # strategy = tf.distribute.experimental.TPUStrategy(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        gpus = tf.config.list_physical_devices('GPU')
        num_gpus = len(gpus)

        if num_gpus:
            if num_gpus > 1:
                logger.info(f"Multiple GPUs are available, using MirroredStrategy. Count: {num_gpus}")
                strategy = tf.distribute.MirroredStrategy()
            else:
                logger.info("One GPU is available, using OneDeviceStrategy on GPU.")
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            logger.info("No GPUs found, using default strategy (likely on CPU).")
            strategy = tf.distribute.get_strategy()
            
    return strategy, is_tpu