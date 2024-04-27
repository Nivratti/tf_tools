import keras
from keras import layers
import tensorflow as tf

class CAM(layers.Layer):
    """
    Channel Attention Module (CAM) enhances specific features across channel dimensions
    by applying channel-wise attention mechanisms.

    Attributes:
        scale_gamma_initializer: Initializer for the scaling factor gamma.
        scale_gamma_regularizer: Optional regularizer for the gamma weight.
        scale_gamma_constraint: Optional constraint for the gamma weight.
        activation_func: Name of the activation function to apply after computing attention scores ('sigmoid', 'softmax', etc.).
    """
    def __init__(self, scale_gamma_initializer='zeros', scale_gamma_regularizer=None, scale_gamma_constraint=None, activation_func='sigmoid', **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.scale_gamma_initializer = scale_gamma_initializer
        self.scale_gamma_regularizer = scale_gamma_regularizer
        self.scale_gamma_constraint = scale_gamma_constraint
        self.activation_func = activation_func

    def build(self, input_shape):
        self.scale_gamma = self.add_weight(
            shape=(1,),
            initializer=self.scale_gamma_initializer,
            name='scale_gamma',
            regularizer=self.scale_gamma_regularizer,
            constraint=self.scale_gamma_constraint
        )
        super(CAM, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        # Extract dynamic dimensions of the input tensor
        batch_size, height, width, num_filters = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]

        # Flatten the spatial dimensions and create a matrix of shape (batch_size, height*width, num_filters)
        flattened_features = tf.reshape(inputs, (batch_size, height * width, num_filters))
        transposed_features = tf.transpose(flattened_features, perm=[0, 2, 1])

        # Compute the matrix multiplication between transposed and original feature matrix
        channel_interaction = tf.matmul(transposed_features, flattened_features)

        # Apply the selected activation function to obtain attention scores
        if self.activation_func == 'softmax':
            attention_scores = layers.Softmax(axis=-1)(channel_interaction)
        elif self.activation_func == 'sigmoid':
            attention_scores = layers.Activation('sigmoid')(channel_interaction)
        else:
            raise ValueError(f"Unsupported activation function '{self.activation_func}'. Choose 'softmax', 'sigmoid', or implement additional activations.")

        # Use the attention scores to scale the original feature matrix
        scaled_features = tf.matmul(flattened_features, attention_scores)
        reshaped_features = tf.reshape(scaled_features, (batch_size, height, width, num_filters))

        # Scale the attention-enhanced output by gamma and add back the input
        output = self.scale_gamma * reshaped_features + inputs
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
