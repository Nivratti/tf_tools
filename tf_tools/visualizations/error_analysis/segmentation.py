import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def display_predictions(model, dataset, num_images=4, verbose=False):
    """
    Displays a comparison between actual masks and predicted masks for a given dataset.

    This function processes a batch of images and their corresponding masks from the dataset,
    performs predictions using the provided model, and visualizes the input images, the true masks,
    and the predicted masks side-by-side.

    Parameters:
    - model (tf.keras.Model): The trained model used for making predictions.
    - dataset (tf.data.Dataset): A dataset object that yields batches of (images, masks) pairs.
                                The dataset should provide data in the format that the model expects.
    - num_images (int, optional): The number of images to display from the dataset. Default is 4.
    - verbose (bool, optional): If True, prints detailed information about the shapes and data types
                               of the images, masks, and predictions. Default is False.

    Usage Example:
    # Assuming 'test_ds' is a TensorFlow dataset formatted correctly (images, masks)
    try:
        display_predictions(model, test_ds, num_images=8, verbose=True)
    except Exception as e:
        print(f"Error: {e}")
    """
    for images, masks in dataset.take(1):
        images = images[:num_images]
        masks = masks[:num_images]
        
        ## TODO: check predict and predict_on_batch not working correctly
        # preds = model.predict(images, verbose=1) 
        # preds = model.predict_on_batch(images)
        preds = model(images) # You can directly call the model on your input data. This is essentially the same as using model.predict() for a single batch but is a more "Pythonic" way of using models in TensorFlow.

        # for debugging
        if verbose:
            print("Image dtype and shape:", images.dtype, images.shape)
            print("Mask dtype and shape:", masks.dtype, masks.shape)
            print("Prediction dtype and shape:", preds.dtype, preds.shape)

        # Dynamic dimensions based on content
        columns = 3  # For input image, true mask, and predicted mask
        width_per_column = 5  # Assuming 5 inches is sufficient for each column
        height_per_image = 2  # Height per image row
        
        # Calculate total width and height
        total_width = width_per_column * columns
        total_height = height_per_image * num_images

        plt.figure(figsize=(total_width, total_height))

        for i in range(num_images):
            # Convert images to a displayable format
            image_display = images[i].numpy() if hasattr(images[i], 'numpy') else images[i]
            if image_display.dtype != np.uint8:
                image_display = (image_display * 255).astype(np.uint8)
            
            # Convert masks to a displayable format
            mask_display = masks[i].numpy() if hasattr(masks[i], 'numpy') else masks[i]
            if mask_display.ndim == 3:
                mask_display = mask_display[:, :, 0]  # Assuming mask is single-channel
            if mask_display.dtype != np.uint8:
                mask_display = (mask_display * 255).astype(np.uint8)
            
            # Convert predictions to a displayable format
            pred_display = np.squeeze(preds[i])
            if pred_display.ndim == 3:
                pred_display = pred_display[:, :, 0]  # Assuming prediction is single-channel
            if pred_display.dtype != np.uint8:
                pred_display = (pred_display * 255).astype(np.uint8)

            plt.subplot(num_images, 3, i*3 + 1)
            plt.imshow(image_display)
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(num_images, 3, i*3 + 2)
            plt.imshow(mask_display, cmap='gray')
            plt.title("True Mask")
            plt.axis('off')

            plt.subplot(num_images, 3, i*3 + 3)
            plt.imshow(pred_display, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        break  # Only display the first batch (or specified number of images)

def display_error_heatmaps(model, dataset, num_images=4):
    """
    Displays error heatmaps for comparing actual masks to predicted masks using a given dataset.

    This function processes a batch of images and their corresponding true masks from the dataset,
    performs predictions with the provided model, and visualizes the input images, the true masks,
    and the error heatmaps. Error heatmaps are generated by calculating the absolute difference
    between the true masks and predicted masks, highlighting areas of mismatch.

    Parameters:
    - model (tf.keras.Model): The trained model used for generating predictions.
    - dataset (tf.data.Dataset): A dataset object that yields batches of (images, masks) pairs.
                                Ensure the dataset provides data in the format expected by the model.
    - num_images (int, optional): The number of images to display from the dataset. Default is 4.

    Usage Example:
    # Replace 'test_ds' with your actual dataset
    # Ensure 'test_ds' yields (image, mask) pairs and that images and masks are properly formatted
    display_error_heatmaps(model, test_ds, num_images=6)
    """
    for images, true_masks in dataset.take(1):
        images = images[:num_images]
        true_masks = true_masks[:num_images]

        # Perform predictions
        preds_masks = model(images)  # Directly calling the model which is equivalent to model.predict()

        # Setup for visualization
        columns = 3  # For input image, true mask, and error heatmap
        width_per_column = 5
        height_per_image = 2
        total_width = width_per_column * columns
        total_height = height_per_image * num_images

        plt.figure(figsize=(total_width, total_height))

        for i in range(num_images):
            # Convert TensorFlow tensors to numpy arrays for visualization
            image_display = images[i].numpy() if isinstance(images[i], tf.Tensor) else images[i]
            true_mask_display = true_masks[i].numpy() if isinstance(true_masks[i], tf.Tensor) else true_masks[i]
            pred_mask_display = preds_masks[i].numpy() if isinstance(preds_masks[i], tf.Tensor) else preds_masks[i]

            # Ensure images are in the correct format
            if image_display.dtype != np.uint8:
                image_display = (image_display * 255).astype(np.uint8)
            if true_mask_display.ndim == 3:
                true_mask_display = true_mask_display[:, :, 0]
            if true_mask_display.dtype != np.uint8:
                true_mask_display = (true_mask_display * 255).astype(np.uint8)
            if pred_mask_display.ndim == 3:
                pred_mask_display = pred_mask_display[:, :, 0]
            if pred_mask_display.dtype != np.uint8:
                pred_mask_display = (pred_mask_display * 255).astype(np.uint8)

            # Calculate absolute error for error heatmap
            error = np.abs(true_mask_display.astype(np.float32) - pred_mask_display.astype(np.float32))
            error = (error * 255).astype(np.uint8) if error.dtype != np.uint8 else error

            plt.subplot(num_images, 3, i * 3 + 1)
            plt.imshow(image_display)
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(num_images, 3, i * 3 + 2)
            plt.imshow(true_mask_display, cmap='gray', vmin=0, vmax=255)
            plt.title("True Mask")
            plt.axis('off')

            plt.subplot(num_images, 3, i * 3 + 3)
            plt.imshow(error, cmap='hot', vmin=0, vmax=255)
            plt.title("Error Heatmap")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        break  # Only display the first batch (or specified number of images)
