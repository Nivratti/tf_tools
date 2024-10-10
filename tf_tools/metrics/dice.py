import tensorflow as tf
from tf_keras.metrics import Metric

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
        """
        Initialize the DiceCoefficient metric.

        Parameters:
        - name: Name of the metric.
        - smooth: A small constant to avoid division by zero.
        """
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the metric with the current batch.

        Parameters:
        - y_true: Ground truth tensor.
        - y_pred: Predicted tensor.
        - sample_weight: Optional weight for each sample.
        """
        # Flatten the tensors to 1D
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        # Compute intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        # Update the state variables
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        """
        Calculate and return the Dice coefficient.
        """
        return (2. * self.intersection + self.smooth) / (self.union + self.smooth)

    def reset_states(self):
        """
        Reset the metric state variables.
        """
        self.intersection.assign(0.)
        self.union.assign(0.)
