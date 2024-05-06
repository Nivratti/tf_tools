import tensorflow as tf
from tf_keras.metrics import Metric
from tf_keras.backend import epsilon

class F1Score(Metric):
    def __init__(self, threshold=0.5, name='f1_score', **kwargs):
        """Initializes the F1 Score metric.

        Args:
            threshold (float): Threshold value for binary classification.
            name (str): Name of the metric.
            **kwargs: Additional keyword arguments passed to the superclass.
        """
        super(F1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the true positive, false positive, and false negative counts.

        Args:
            y_true (Tensor): The ground truth labels.
            y_pred (Tensor): The predicted values.
            sample_weight (Tensor): Optional tensor of weights for weighted calculation.
        """
        # Convert y_pred based on the threshold
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # Calculate true positives, false positives, and false negatives
        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

        # Update state variables
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        """Calculates and returns the F1 score."""
        precision = self.true_positives / (self.true_positives + self.false_positives + epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + epsilon())
        f1_score = 2 * ((precision * recall) / (precision + recall + epsilon()))
        return f1_score

    def reset_state(self):
        """Resets all of the metric state variables."""
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
