import tensorflow as tf
from tensorflow.keras.layers import Layer, Activation


class CAM(Layer):
    """
    Channel Attention Module (CAM) implements a channel-wise attention mechanism 
    to capture inter-dependencies among channels in feature maps from convolutional layers. 
    This module uses a self-attention mechanism where the attention is computed based on 
    the channel-wise relationships within the input feature map.
    
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
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        """
        Build the internal components of the layer based on the input shape.
        This method initializes the weights of the layer based on the input shape.
        
        Args:
            input_shape: Shape tuple of the input feature maps.
        """
        # Initialize gamma, a trainable weight for the output
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        super(CAM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        The logic of the layer's forward pass, which computes the output based on the 
        input tensors and the layer's parameters.
        
        Args:
            inputs: Input tensor, the feature map from a convolutional layer.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Tensor of the same shape as `inputs`, representing the attention-weighted
            feature map.
        """
        # Flatten the spatial dimensions of the input for channel-wise attention
        vec_a = tf.reshape(inputs, [-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]])

        # Compute the channel-wise attention map
        vec_aT = tf.transpose(vec_a, perm=[0, 2, 1])
        aTa = tf.matmul(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)

        # Apply the attention map to the input features
        aaTa = tf.matmul(softmax_aTa, vec_aT)
        aaTa = tf.transpose(aaTa, perm=[0, 2, 1])
        aaTa = tf.reshape(aaTa, tf.shape(inputs))

        # Output is a weighted sum of the input and the attention-weighted feature map
        out = self.gamma * aaTa + inputs
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
    

class CAM_DynamicShape(Layer):
    """
    DynamicCAM is an enhanced Channel Attention Module designed to dynamically adapt to varying input shapes.
    If we pass None to model shape and try to build model with this layer it will work fine. 
    """
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        """
        Build the internal components of the layer based on the input shape.
        This method initializes the weights of the layer based on the input shape.
        
        Args:
            input_shape: Shape tuple of the input feature maps.
        """
        # Initialize gamma, a trainable weight for the output
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        super(CAM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        The logic of the layer's forward pass, which computes the output based on the 
        input tensors and the layer's parameters.
        
        Args:
            inputs: Input tensor, the feature map from a convolutional layer.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Tensor of the same shape as `inputs`, representing the attention-weighted
            feature map.
        """
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        # Reshape inputs dynamically
        vec_a = tf.reshape(inputs, [batch_size, height * width, channels])

        # Compute the channel-wise attention map
        vec_aT = tf.transpose(vec_a, perm=[0, 2, 1])
        aTa = tf.matmul(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)

        # Apply the attention map to the input features
        aaTa = tf.matmul(softmax_aTa, vec_aT)
        aaTa = tf.transpose(aaTa, perm=[0, 2, 1])
        
        # Reshape back to the original input shape dynamically
        aaTa = tf.reshape(aaTa, [batch_size, height, width, channels])

        # Output is a weighted sum of the input and the attention-weighted feature map
        out = self.gamma * aaTa + inputs
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