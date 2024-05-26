import matplotlib.pyplot as plt
import numpy as np

def visualize_cls_dataset_in_grid(dataset, label_to_id_mapping, num_samples=6, images_per_row=3):
    """
    Visualizes a few samples from the given TensorFlow dataset, displaying each image and its label in a grid format.

    Args:
    - dataset (tf.data.Dataset): The TensorFlow dataset to visualize.
    - label_to_id_mapping (dict): Mapping from label IDs to their string representations.
    - num_samples (int): Number of samples to visualize.
    - images_per_row (int): Number of images to display per row.
    """
    # Take a single batch from the dataset
    for images, labels in dataset.take(1):
        num_rows = (num_samples + images_per_row - 1) // images_per_row  # Calculate number of rows needed

        plt.figure(figsize=(images_per_row * 4, num_rows * 4))  # Adjust the size based on the number of samples
        plt.subplots_adjust(wspace=0.2, hspace=0.5)  # Adjust the spacing between subplots

        for i in range(num_samples):
            ax_image = plt.subplot(num_rows, images_per_row, i + 1)
            # Denormalize the image if necessary
            denormalized_image = images[i].numpy() * 255
            denormalized_image = denormalized_image.astype('uint8')
            plt.imshow(denormalized_image)
            plt.title(f"Label: {label_to_id_mapping[labels[i].numpy()]}")
            plt.axis("off")

        plt.show()
        break  # Only display the first batch
