import tensorflow as tf

def check_mask_values(masks):
    """
    Checks and prints the unique values present in a given mask tensor. 
    It specifically determines whether the mask is binary, containing only 0 and 255.

    Args:
    - masks (tf.Tensor): A mask tensor of any shape.

    This function flattens the mask tensor, calculates the unique values,
    and checks if those values are only 0 and 255, indicating a binary mask.
    """
    # Flatten the mask tensor to ensure all values across the masks are considered
    flat_masks = tf.reshape(masks, [-1])

    # Find the unique values in the flattened mask
    unique_values, _ = tf.unique(flat_masks)

    # Convert the unique values tensor to a numpy array for easier handling
    unique_values_np = unique_values.numpy()

    # Print the unique values found in the mask
    print("Unique values in the mask:", unique_values_np)

    # Check if the mask contains exactly two unique values
    if len(unique_values_np) == 2:
        print("Mask contains only two values.")
    else:
        print(f"Mask does not contain only two values. It contains {len(unique_values_np)} total unique values.")
