from tf_tools.metrics.dice import dice_coefficient

def dice_loss(y_true, y_pred):
    """
    Dice loss for binary masks.
    
    Parameters:
    - y_true: The ground truth tensor.
    - y_pred: The predicted tensor.
    
    Returns:
    - loss: The Dice loss.
    
    Usage:
    model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient])
    """
    return 1 - dice_coefficient(y_true, y_pred)