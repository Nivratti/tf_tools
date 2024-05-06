import tensorflow as tf
from tf_keras.layers import Layer, Conv1D


class EfficientChannelAttention2D(Layer):
    """
    EfficientChannelAttention2D implements an efficient channel attention mechanism for convolutional networks in TensorFlow.
    This layer computes channel-wise attention by applying global average pooling followed by a Conv1D layer to generate attention scores,
    which are then applied to the input feature maps to emphasize important channels.
    
    Args:
        nf (int): Number of filters or channels in the Conv2D feature maps.
        **kwargs: Additional keyword arguments for the Layer superclass.
    """

    def __init__(self, nf, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.conv1 = None  # Placeholder for the Conv1D layer, to be defined in build

    def build(self, input_shape):
        """
        Initializes the internal Conv1D layer based on the input shape.
        This method dynamically sets up the Conv1D layer to operate on the channel dimension of the input feature maps.
        
        Args:
            input_shape: Shape tuple of the input feature maps.
        """
        # Initialize the Conv1D layer with 1 filter, kernel size of 3, and 'same' padding
        self.conv1 = Conv1D(filters=1, kernel_size=3, activation=None, padding="same", use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        """
        Forward pass of the layer. Applies efficient channel attention to the input feature maps.
        
        Args:
            inputs: Input tensor, the feature map from a convolutional layer.
            
        Returns:
            Tensor of the same shape as `inputs`, representing the channel-wise attended feature map.
        """
        # Global average pooling across the spatial dimensions to reduce each channel to a single value
        pool = tf.reduce_mean(inputs, [1, 2])
        pool = tf.expand_dims(pool, -1)  # Add an extra dimension for compatibility with Conv1D

        # Apply Conv1D to the pooled output to generate channel-wise attention scores
        att = self.conv1(pool)
        att = tf.transpose(att, perm=[0, 2, 1])  # Transpose to match the channel dimension
        att = tf.expand_dims(att, 1)  # Add spatial dimensions back
        att = tf.sigmoid(att)  # Apply sigmoid to get attention weights in [0, 1]

        # Multiply original input by the attention weights to emphasize important channels
        y = tf.multiply(inputs, att)
        return y

    def get_config(self):
        """
        Returns the configuration of the layer as a dictionary.
        This method supports the functionality of saving and loading models.
        
        Returns:
            A dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update({"nf": self.nf})
        return config
    