import tensorflow as tf
from keras import layers
from tensorflow.keras import backend as K

class PAM(layers.Layer):
    """
    Position Attention Module (PAM) as a custom Keras Layer.

    Captures spatial attention mechanisms within a feature map by projecting the input
    into different spaces to calculate attention, then scales the input feature map by
    the attention scores to enhance features with inter-spatial relevance.

    Attributes:
        scale_gamma_initializer: Initializer for the scaling factor.
        scale_gamma_regularizer: Regularizer for the scaling factor.
        scale_gamma_constraint: Constraint for the scaling factor.
        activation_func: The type of activation function to use ('softmax' or 'sigmoid').
    """
    def __init__(self, scale_gamma_initializer='zeros', scale_gamma_regularizer=None, scale_gamma_constraint=None, activation_func='sigmoid', **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.scale_gamma_initializer = scale_gamma_initializer
        self.scale_gamma_regularizer = scale_gamma_regularizer
        self.scale_gamma_constraint = scale_gamma_constraint
        self.activation_func = activation_func

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError("The channel dimension of the inputs should be defined. Found `None`.")

        self.scale_gamma = self.add_weight(
            name='scale_gamma', 
            shape=(1,),
            initializer=self.scale_gamma_initializer,
            regularizer=self.scale_gamma_regularizer,
            constraint=self.scale_gamma_constraint
        )

        num_channels = input_shape[-1]
        self.query_conv = layers.Conv2D(num_channels // 8, 1, use_bias=False, kernel_initializer='he_normal')
        self.key_conv = layers.Conv2D(num_channels // 8, 1, use_bias=False, kernel_initializer='he_normal')
        self.value_conv = layers.Conv2D(num_channels, 1, use_bias=False, kernel_initializer='he_normal')

        # Build the internal Conv2D layers with the correct input shape
        self.query_conv.build(input_shape)
        self.key_conv.build(input_shape)
        self.value_conv.build(input_shape)
        super(PAM, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        # Extract dimensions to handle dynamic shape scenarios
        shape = tf.shape(inputs)
        batch_size, height, width, num_filters = shape[0], shape[1], shape[2], shape[3]

        # Generate query and key features
        query_features = self.query_conv(inputs)
        key_features = self.key_conv(inputs)

        # Prepare for matrix multiplication by reshaping query and key features
        query_flat = K.reshape(query_features, (batch_size, height * width, num_filters // 8))
        key_flat_transposed = K.permute_dimensions(K.reshape(key_features, (batch_size, height * width, num_filters // 8)), (0, 2, 1))

        # Compute attention scores using matrix multiplication
        attention_scores = K.batch_dot(query_flat, key_flat_transposed)
        
        # Apply the specified activation function to the attention scores
        if self.activation_func == 'softmax':
            attention_scores = layers.Softmax(axis=-1)(attention_scores)
        elif self.activation_func == 'sigmoid':
            attention_scores = layers.Activation('sigmoid')(attention_scores)
        else:
            raise ValueError(f"Unsupported activation function '{self.activation_func}'. Choose 'softmax' or 'sigmoid'.")

        # Compute output features by applying the attention scores to the value features
        value_features = self.value_conv(inputs) # Convolution layer to transform the input for output scaling
        value_flat = K.reshape(value_features, (batch_size, height * width, num_filters))
        attended_features = K.batch_dot(attention_scores, value_flat)
        attended_features_reshaped = K.reshape(attended_features, (batch_size, height, width, num_filters))

        # Scale the output by the learned gamma factor and add the input
        output = self.scale_gamma * attended_features_reshaped + inputs
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
