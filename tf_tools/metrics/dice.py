import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice coefficient between two tensors.
    
    Parameters:
    - y_true: The ground truth tensor.
    - y_pred: The predicted tensor.
    - smooth: A small constant to avoid division by zero.
    
    Returns:
    - dice: The Dice coefficient.
    """
    # Ensure both tensors are of the same float32 data type
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    # Compute the intersection and the sum of the two sets
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    # Compute the Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice