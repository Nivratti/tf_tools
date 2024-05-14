import tensorflow as tf
from tf_keras.metrics import Metric
from tf_keras.backend import epsilon

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


class IoUCoefficient(Metric):
    """
    Keras metric to use as class and use keras like methods like update_state
    """
    def __init__(self, name='iou_coefficient', smooth=1e-6, **kwargs):
        super(IoUCoefficient, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.iou = self.add_weight(name="iou", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Directly update the IoU value
        self.iou.assign(iou)

    def result(self):
        # return self.iou
        return self.iou + epsilon()

    def reset_states(self):
        # Reset the accumulated value.
        self.iou.assign(0.0)


class BinaryIoU(Metric):
    def __init__(self, target_class_ids=(0, 1), threshold=0.5, name='binary_iou', **kwargs):
        super(BinaryIoU, self).__init__(name=name, **kwargs)
        self.target_class_ids = target_class_ids
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.sigmoid(y_pred)  # Convert logits to probabilities
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        for class_id in self.target_class_ids:
            # Calculate true positives, false positives, and false negatives for each class
            tp = tf.reduce_sum(tf.cast((y_pred == class_id) & (y_true == class_id), tf.float32))
            fp = tf.reduce_sum(tf.cast((y_pred == class_id) & (y_true != class_id), tf.float32))
            fn = tf.reduce_sum(tf.cast((y_pred != class_id) & (y_true == class_id), tf.float32))

            self.true_positives.assign_add(tp)
            self.false_positives.assign_add(fp)
            self.false_negatives.assign_add(fn)

    def result(self):
        iou = self.true_positives / (self.true_positives + self.false_positives + self.false_negatives + 1e-7)
        return iou

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
