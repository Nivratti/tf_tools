import tensorflow as tf

def iou_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute the Intersection over Union (IoU) between two tensors.
    
    Parameters:
    - y_true: The ground truth tensor.
    - y_pred: The predicted tensor.
    - smooth: A small constant to avoid division by zero.
    
    Returns:
    - iou: The IoU coefficient.
    
    Method:
    model.compile(optimizer='adam', loss=iou_loss, metrics=[iou_coefficient])
    """
    # Flatten the tensors to [batch_size, -1]
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    # Compute the intersection and the sum of the two sets
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    # Compute the IoU coefficient
    iou = (intersection + smooth) / (union + smooth)
    return iou


class IoUCoefficient(tf.keras.metrics.Metric):
    """
    Keras metric to use as class and use keras like methods like update_state
    """
    def __init__(self, name='iou_coefficient', smooth=1e-6, **kwargs):
        """Metric to calculate IoU coefficient."""
        super(IoUCoefficient, self).__init__(name=name, **kwargs)
        self.smooth = 1e-6

    def update_state(self, y_true, y_pred, sample_weight=None):
        iou = iou_coefficient(y_true, y_pred, self.smooth)
        self.iou = iou  # Store the current batch's dice score

    def result(self):
        return self.iou

    def reset_state(self):
        pass  # State does not accumulate from batch to batch

