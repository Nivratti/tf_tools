import numpy as np
import matplotlib.pyplot as plt

def display_predictions(model, dataset, num_images=4):
    """
    Display a comparison between actual and predicted masks for a given dataset.

    Parameters:
    - model: The trained model to use for predictions.
    - dataset: The dataset to use for generating predictions. Expects a batch of (images, masks).
    - num_images: Optional. Number of images to display from the dataset.

    Usage:
    # Replace 'test_ds' with your actual dataset
    # Ensure 'test_ds' yields (image, mask) pairs and that images are normalized as expected by your model
    try:
        display_predictions(model, test_ds, num_images=8)
    except Exception as e:
        print(f"Error: {e}")
    """
    for images, masks in dataset.take(1):
        images = images[:num_images]
        masks = masks[:num_images]

        preds = model.predict(images, verbose=1)

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
    Display error heatmaps for comparing actual and predicted masks.

    Parameters:
    - model: The trained model to use for predictions.
    - dataset: The dataset to use for generating predictions. Expects a batch of (images, masks).
    - num_images: Optional. Number of images to display from the dataset.
    """
    for images, true_masks in dataset.take(1):
        images = images[:num_images]
        true_masks = true_masks[:num_images]

        preds_masks = model.predict(images)

        # Dynamic dimensions based on content
        columns = 3  # For input image, true mask, and error heatmap
        width_per_column = 5  # Width for each column
        height_per_image = 2  # Height for each row of images

        # Calculate total width and height for the figure
        total_width = width_per_column * columns
        total_height = height_per_image * num_images
        
        plt.figure(figsize=(total_width, total_height))
        
        for i in range(num_images):
            # Convert images to a displayable format
            image_display = images[i].numpy() if hasattr(images[i], 'numpy') else images[i]
            if image_display.dtype != np.uint8:
                image_display = (image_display * 255).astype(np.uint8)
            
            # Convert true masks to a displayable format
            true_mask_display = true_masks[i].numpy() if hasattr(true_masks[i], 'numpy') else true_masks[i]
            if true_mask_display.ndim == 3:
                true_mask_display = true_mask_display[:, :, 0]
            if true_mask_display.dtype != np.uint8:
                true_mask_display = (true_mask_display * 255).astype(np.uint8)
            
            # Process predictions
            pred_mask_display = preds_masks[i]
            if pred_mask_display.ndim == 3:
                pred_mask_display = pred_mask_display[:, :, 0]
            if pred_mask_display.dtype != np.uint8:
                pred_mask_display = (pred_mask_display * 255).astype(np.uint8)
            
            # Calculate absolute error for error heatmap
            error = np.abs(true_mask_display.astype(np.float32) - pred_mask_display.astype(np.float32))
            error = (error * 255).astype(np.uint8) if error.dtype != np.uint8 else error
            
            plt.subplot(num_images, 3, i*3 + 1)
            plt.imshow(image_display)
            plt.title("Input Image")
            plt.axis('off')
            
            plt.subplot(num_images, 3, i*3 + 2)
            plt.imshow(true_mask_display, cmap='gray', vmin=0, vmax=255)
            plt.title("True Mask")
            plt.axis('off')
            
            plt.subplot(num_images, 3, i*3 + 3)
            plt.imshow(error, cmap='hot', vmin=0, vmax=255)
            plt.title("Error Heatmap")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        break  # Only display the first batch (or specified number of images)