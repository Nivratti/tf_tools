import tensorflow as tf
from keras.metrics import Metric

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


class DiceCoefficient(Metric):
    def __init__(self, name='dice_coefficient', smooth=1e-6, **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        # Use add_weight to correctly manage the state
        self.intersect = self.add_weight(name="intersect", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        # Update the state variables in a way that's compatible with TensorFlow's execution
        self.intersect.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        # Compute the Dice coefficient using the state variables
        dice = (2. * self.intersect + self.smooth) / (self.union + self.smooth)
        return dice

    def reset_states(self):
        # Reset the state of the metric
        self.intersect.assign(0)
        self.union.assign(0)
