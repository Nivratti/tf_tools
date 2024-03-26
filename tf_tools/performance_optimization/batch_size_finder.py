import tensorflow as tf

def find_max_batch_size(model, prepare_dataset_fn, training_tfrecord_filepaths, config):
    """
    Finds the maximum batch size that can be processed by a accelerator such as GPU or TPU for a given TensorFlow model by dynamically adjusting the batch size and monitoring for out-of-memory (OOM) errors.

    Args:
        model (keras.Model): The TensorFlow model for which to find the max batch size.
        prepare_dataset_fn (Callable): A function that prepares a TensorFlow dataset for training. It should accept file paths to TFRecord files, target image size, batch size, and whether to maintain aspect ratio as inputs, and return a tf.data.Dataset.
        training_tfrecord_filepaths (list of str): List of file paths to the training TFRecord files.
        config (object): A configuration object that must contain `image_height`, `image_width`, `should_maintain_aspect_ratio` attributes to specify the image dimensions and whether to maintain aspect ratio.

    Returns:
        int: The maximum batch size that can be processed without encountering OOM error.

    This function attempts to fit the model using the training data prepared by `prepare_dataset_fn`, starting with a batch size of 1 and doubling it until an OOM error occurs. The largest successful batch size is then returned.
    """
    batch_size = 1  # Starting batch size
    max_successful_batch_size = batch_size  # Track the largest successful batch size

    while True:
        try:
            # # Generate dummy data for the current batch size
            # dummy_images, dummy_masks = generate_dummy_data(image_shape, batch_size)

            train_ds = prepare_dataset_fn(
                training_tfrecord_filepaths,
                target_size=(config.image_height, config.image_width),
                batch_size=batch_size, # config.batch_size,
                maintain_aspect_ratio=config.should_maintain_aspect_ratio,
            )

            # Attempt to train the model with the current batch size
            print(f"Trying batch size: {batch_size}")
            model.fit(train_ds, epochs=1, steps_per_epoch=1, verbose=0) # verbose=2)

            # If successful, double the batch size
            max_successful_batch_size = batch_size
            batch_size *= 2
        except tf.errors.ResourceExhaustedError:
            print(f"Batch size of {batch_size} is too large. Maximum successful batch size: {max_successful_batch_size}")
            break  # Exit the loop if OOM error occurs

    return max_successful_batch_size