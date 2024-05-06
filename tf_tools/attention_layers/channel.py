import tensorflow as tf
from tf_keras.layers import Layer, Conv2D


class ChannelAttention2D(Layer):
    """
    Implements Channel Attention as described by Sanghyun Woo et al. for convolutional networks in TensorFlow.
    This layer enhances Conv2D feature maps by focusing on 'informative' channels through an attention mechanism.
    
    Args:
        nf (int): Number of filters or channels in the Conv2D feature maps.
        r (int): Reduction factor to control the bottleneck feature's channel dimension.
        **kwargs: Additional keyword arguments for the Layer superclass.

    reference:
        https://github.com/vinayak19th/Visual_attention_tf
    """
    def __init__(self, nf, r=4, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf  # Number of filters in the feature map
        self.r = r  # Reduction factor
        # Placeholder for convolution layers to be defined in `build`
        self.conv1 = None
        self.conv2 = None

    def build(self, input_shape):
        """
        Initializes the Conv2D layers with the appropriate shapes based on the input shape.
        This method dynamically sets up the layers to operate on the channel dimension of the input feature maps.
        
        Args:
            input_shape: Shape tuple of the input feature maps.
        """
        # First Conv2D layer reduces channel dimension by the reduction factor `r`
        self.conv1 = Conv2D(filters=self.nf // self.r, kernel_size=1, use_bias=True)
        # Second Conv2D layer restores channel dimension to original number of filters `nf`
        self.conv2 = Conv2D(filters=self.nf, kernel_size=1, use_bias=True)
        super().build(input_shape)

    def call(self, inputs):
        """
        Forward pass of the layer. Applies channel attention to the input feature maps.
        
        Args:
            inputs: Input tensor, the feature map from a convolutional layer.
            
        Returns:
            Tensor of the same shape as `inputs`, representing the attention-weighted feature map.
        """
        # Global average pooling across the spatial dimensions to reduce each channel to a single value
        y = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        # Apply a bottleneck Conv2D layer to reduce dimensionality (channel-wise)
        y = self.conv1(y)
        # ReLU activation for introducing non-linearity
        y = tf.nn.relu(y)
        # Expand the channel dimension back to the original using another Conv2D layer
        y = self.conv2(y)
        # Apply sigmoid activation to get attention weights in [0, 1]
        y = tf.nn.sigmoid(y)
        # Multiply original input by the attention weights to emphasize important channels
        return tf.multiply(inputs, y)

    def get_config(self):
        """
        Returns the configuration of the layer as a dictionary for model serialization.
        
        Returns:
            A dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update({"nf": self.nf, "r": self.r})
        return config