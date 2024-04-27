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


def test_pam_shape(input_shape):
    """
    Tests whether the PAM layer retains the same input and output shape.
    
    Args:
    input_shape (tuple): The shape of the input tensor to test, excluding batch size.
    
    Returns:
    bool: True if the shape test passes, False otherwise.
    """
    # Create a random tensor with the specified shape
    input_tensor = tf.random.normal([1] + list(input_shape))

    # Initialize the PAM layer
    pam_layer = PAM()

    # Get the output from the PAM layer
    output_tensor = pam_layer(input_tensor)

    # Check if the output shape matches the input shape
    if input_tensor.shape == output_tensor.shape:
        print("Shape Test Passed: Input shape and output shape are the same.")
        return True
    else:
        print("Shape Test Failed: Output shape does not match input shape.")
        return False

def create_patterned_input(input_shape, pattern_size=(2, 2)):
    """
    Creates a test input tensor with zeros and a specific pattern in the center.
    
    Args:
    input_shape (tuple): The shape of the input tensor, excluding batch size.
    pattern_size (tuple): The size of the square pattern to be placed in the center.
    
    Returns:
    tf.Tensor: The created input tensor with a pattern.
    """
    # Full zeros tensor
    base_input = tf.zeros([1] + list(input_shape), dtype=tf.float32)
    
    # Calculate the start indices for the pattern
    start_idx_h = input_shape[0] // 2 - pattern_size[0] // 2
    start_idx_w = input_shape[1] // 2 - pattern_size[1] // 2

    # Create indices for updates
    indices = []
    updates = []
    for i in range(pattern_size[0]):
        for j in range(pattern_size[1]):
            for k in range(input_shape[2]):
                indices.append([0, start_idx_h + i, start_idx_w + j, k])
                updates.append(1.0)

    # Convert lists to tensors
    indices = tf.constant(indices, dtype=tf.int32)
    updates = tf.constant(updates, dtype=tf.float32)

    # Update the base input tensor with a pattern
    pattern_input = tf.tensor_scatter_nd_update(base_input, indices, updates)
    return pattern_input

def test_pam_functionality(input_shape):
    """
    Tests the functionality of the PAM layer to ensure it alters the input tensor.
    
    Args:
    input_shape (tuple): The shape of the input tensor to test, excluding batch size.
    
    Returns:
    bool: True if the functionality test passes, False otherwise.
    """
    # Create a controlled test input with a specific pattern
    test_input = create_patterned_input(input_shape, pattern_size=(1, 1))

    # Initialize the PAM layer
    pam_layer = PAM(scale_gamma_initializer='ones')

    # Get the output from the PAM layer
    output_tensor = pam_layer(test_input)

    # Calculate mean of the input and output tensors
    input_mean = tf.reduce_mean(test_input)
    output_mean = tf.reduce_mean(output_tensor)

    # Check if there is a statistical difference between the input and output
    if not tf.math.equal(input_mean, output_mean):
        print("Functionality Test Passed: Output is statistically different from the input.")
        return True
    else:
        print("Functionality Test Failed: Output is statistically the same as the input.")
        return False

if __name__ == "__main__":
    from keras.models import Model
    from keras.layers import Input
    import keras

    # Define the input tensor
    # input_tensor = Input(shape=(64, 640, 1408))
    input_tensor = Input(shape=(None, None, 1408))

    # Create an instance of your custom layer
    attention_layer = PAM()

    # Apply your custom layer
    output_tensor = attention_layer(input_tensor)

    # Build the model
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # Summary of the model to see all parameters including those from PAM
    print(model.summary())
    
    print(f"\nPerforming unit testing:")
    ## 2. Unit testing
    ### 2.1 shape testing
    test_pam_shape((128, 128, 32))
    
    ### 2.2 Functionality testing
    test_pam_functionality((2, 20, 1408))
