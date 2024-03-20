import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset_in_rows(dataset, num_samples=5):
    """
    Visualizes a few samples from the given TensorFlow dataset, displaying each image and its mask in its own row.

    Args:
    - dataset (tf.data.Dataset): The TensorFlow dataset to visualize.
    - label_to_id_mapping (dict): Mapping from label IDs to their string representations.
    - num_samples (int): Number of samples to visualize.
    """
    # Take a single batch from the dataset
    for images, masks in dataset.take(1):
        plt.figure(figsize=(12, num_samples * 1))  # Adjust the size based on the number of samples
        plt.subplots_adjust(wspace=0.2, hspace=0.5)  # Adjust the spacing between subplots

        for i in range(num_samples):
            # Image
            ax_image = plt.subplot(num_samples, 2, 2 * i + 1)
            # Denormalize the image if necessary
            denormalized_image = images[i].numpy() * 255
            denormalized_image = denormalized_image.astype('uint8')
            plt.imshow(denormalized_image)
            # Find the label corresponding to the highest probability in the one-hot encoded vector
            label_id = np.argmax(masks[i].numpy())
            plt.title("Image")
            plt.axis("off")

            # Mask
            ax_mask = plt.subplot(num_samples, 2, 2 * i + 2)
            denormalized_mask = masks[i].numpy() * 255
            denormalized_mask = denormalized_mask.astype('uint8')
            plt.imshow(denormalized_mask[:, :, 0], cmap='gray')  # Assuming mask is grayscale
            plt.title("Mask")
            plt.axis("off")

        plt.show()
