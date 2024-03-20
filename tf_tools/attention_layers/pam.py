import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Activation

class PAM(Layer):
    """
    Position Attention Module (PAM) implements a position-based attention mechanism 
    to capture long-range dependencies in feature maps from convolutional layers. 
    This module uses a self-attention mechanism where the attention is computed 
    based on the positions of the features in the input feature map.
    
    Args:
        gamma_initializer (initializer): Initializer for the gamma weight.
        gamma_regularizer (regularizer): Regularizer function applied to
                                          the gamma weight.
        gamma_constraint (constraint): Constraint function applied to
                                        the gamma weight.
        **kwargs: Additional keyword arguments for the Layer superclass.
    """

    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint
        # Initialize sub-layers without specifying the filter size
        self.conv_b = Conv2D(1, 1, use_bias=False, kernel_initializer='he_normal')  # For generating B feature
        self.conv_c = Conv2D(1, 1, use_bias=False, kernel_initializer='he_normal')  # For generating C feature
        self.conv_d = Conv2D(1, 1, use_bias=False, kernel_initializer='he_normal')  # For generating D feature

    def build(self, input_shape):
        """
        Build the internal components of the layer based on the input shape.
        This method initializes the weights of the layer based on the input shape.
        
        Args:
            input_shape: Shape tuple of the input feature maps.
        """
        filters = input_shape[-1]
        # Adjust filter sizes based on input shape dynamically
        self.conv_b.filters = filters // 8
        self.conv_c.filters = filters // 8
        self.conv_d.filters = filters
        # Initialize gamma, a trainable weight for the output
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        super(PAM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        The logic of the layer's forward pass, which computes the output 
        based on the input tensors and the layer's parameters.
        
        Args:
            inputs: Input tensor, the feature map from a convolutional layer.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Tensor of the same shape as `inputs`, representing the attention-weighted
            feature map.
        """
        # Generate feature maps B, C, and D using convolution
        b = self.conv_b(inputs)
        c = self.conv_c(inputs)
        d = self.conv_d(inputs)

        # Calculate dimensions for reshaping
        h, w = inputs.shape[1], inputs.shape[2]
        # Reshape and transpose for matrix multiplication
        vec_b = tf.reshape(b, [-1, h * w, b.shape[-1]])
        vec_cT = tf.transpose(tf.reshape(c, [-1, h * w, c.shape[-1]]), perm=[0, 2, 1])
        # Compute attention map as the dot product of B and C features
        bcT = tf.matmul(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        # Multiply attention map with D feature and reshape back to original feature map shape
        vec_d = tf.reshape(d, [-1, h * w, d.shape[-1]])
        bcTd = tf.matmul(softmax_bcT, vec_d)
        bcTd = tf.reshape(bcTd, [-1, h, w, d.shape[-1]])

        # Output is a weighted sum of the input and the attention-weighted feature map
        out = self.gamma * bcTd + inputs
        return out

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer given the input shape.
        
        Args:
            input_shape: Shape tuple of the input feature maps.
            
        Returns:
            A shape tuple representing the output shape of the layer.
        """
        return input_shape