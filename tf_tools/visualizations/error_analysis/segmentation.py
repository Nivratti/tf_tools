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
    # Ensure we only display the specified number of images
    for images, masks in dataset.take(1):  # Take one batch from the dataset
        print(f"images.shape: {images.shape}")
        preds = model.predict(images, verbose=1)
        print(f"preds.shape: {preds.shape}")
        plt.figure(figsize=(10, num_images * 1))

        for i in range(num_images):
            plt.subplot(num_images, 3, i*3 + 1)
            plt.imshow(images[i])
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(num_images, 3, i*3 + 2)
            plt.imshow(masks[i], cmap='gray', vmin=0, vmax=1)
            plt.title("True Mask")
            plt.axis('off')

            plt.subplot(num_images, 3, i*3 + 3)
            pred_mask = np.squeeze(preds[i])  # Remove batch dimension
            plt.imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
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
    # Take one batch from the dataset
    for images, true_masks in dataset.take(1):
        preds_masks = model.predict(images)
        plt.figure(figsize=(15, num_images * 4))
        
        for i in range(num_images):
            error = np.abs(true_masks[i] - preds_masks[i])  # Calculate absolute error
            
            plt.subplot(num_images, 3, i*3 + 1)
            plt.imshow(images[i])
            plt.title("Input Image")
            plt.axis('off')
            
            plt.subplot(num_images, 3, i*3 + 2)
            plt.imshow(true_masks[i], cmap='gray', vmin=0, vmax=1)
            plt.title("True Mask")
            plt.axis('off')
            
            plt.subplot(num_images, 3, i*3 + 3)
            plt.imshow(error.squeeze(), cmap='hot', vmin=0, vmax=1)
            plt.title("Error Heatmap")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        break  # Only display the first batch (or specified number of images)
