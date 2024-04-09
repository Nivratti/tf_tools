from tf_tools.metrics.iou import iou_coefficient

def iou_loss(y_true, y_pred):
    """
    IoU loss for binary masks.
    
    Parameters:
    - y_true: The ground truth tensor.
    - y_pred: The predicted tensor.
    
    Returns:
    - loss: The IoU loss.
    """
    return 1 - iou_coefficient(y_true, y_pred)