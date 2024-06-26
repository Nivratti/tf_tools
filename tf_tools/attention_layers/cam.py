from tf_keras import layers
import tensorflow as tf
from tf_keras.utils import register_keras_serializable

@register_keras_serializable(package='Custom', name='CAM')
class CAM(layers.Layer):
    """
    Channel Attention Module (CAM) enhances specific features across channel dimensions
    by applying channel-wise attention mechanisms using softmax activation. The attention
    mechanism scales feature maps based on learned importance from channel interactions.

    Attributes:
        scale_gamma_initializer: Initializer for the scaling factor gamma.
        scale_gamma_regularizer: Optional regularizer for the gamma weight.
        scale_gamma_constraint: Optional constraint for the gamma weight.
    """
    def __init__(self, scale_gamma_initializer='zeros', scale_gamma_regularizer=None, 
                 scale_gamma_constraint=None, **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.scale_gamma_initializer = scale_gamma_initializer
        self.scale_gamma_regularizer = scale_gamma_regularizer
        self.scale_gamma_constraint = scale_gamma_constraint

    def build(self, input_shape):
        self.scale_gamma = self.add_weight(
            shape=(1,),
            initializer=self.scale_gamma_initializer,
            name='scale_gamma',
            regularizer=self.scale_gamma_regularizer,
            constraint=self.scale_gamma_constraint,
            trainable=True
        )
        super(CAM, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        # Extract dynamic dimensions of the input tensor
        shape = tf.shape(inputs)
        batch_size, height, width, num_filters = shape[0], shape[1], shape[2], shape[3]

        # Flatten the spatial dimensions to create a matrix of shape (batch_size, height*width, num_filters)
        flattened_features = tf.reshape(inputs, (batch_size, height * width, num_filters))
        transposed_features = tf.transpose(flattened_features, perm=[0, 2, 1])

        # Compute the matrix multiplication between transposed and original feature matrix
        channel_interaction = tf.matmul(transposed_features, flattened_features)

        # Apply softmax activation function to obtain attention scores
        attention_scores = tf.nn.softmax(channel_interaction, axis=-1)

        # Use the attention scores to scale the original feature matrix
        scaled_features = tf.matmul(flattened_features, attention_scores)
        reshaped_features = tf.reshape(scaled_features, (batch_size, height, width, num_filters))

        # Scale the attention-enhanced output by gamma and add back the input
        output = self.scale_gamma * reshaped_features + inputs

        # Explicitly delete intermediate tensors to release memory
        del flattened_features, transposed_features, channel_interaction, attention_scores, scaled_features, reshaped_features
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(CAM, self).get_config()
        config.update({
            'scale_gamma_initializer': self.scale_gamma_initializer,
            'scale_gamma_regularizer': self.scale_gamma_regularizer,
            'scale_gamma_constraint': self.scale_gamma_constraint
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def test_cam_shape(input_shape):
    """
    Tests whether the CAM layer retains the same input and output shape.
    
    Args:
    input_shape (tuple): The shape of the input tensor to test, excluding batch size.
    
    Returns:
    bool: True if the shape test passes, False otherwise.
    """
    # Create a random tensor with the specified shape
    input_tensor = tf.random.normal([1] + list(input_shape))

    # Initialize the PAM layer
    cam_layer = CAM()

    # Get the output from the PAM layer
    output_tensor = cam_layer(input_tensor)

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

def test_cam_functionality(input_shape):
    """
    Tests the functionality of the CAM layer to ensure it alters the input tensor.
    
    Args:
    input_shape (tuple): The shape of the input tensor to test, excluding batch size.
    
    Returns:
    bool: True if the functionality test passes, False otherwise.
    """
    # Create a controlled test input with a specific pattern
    test_input = create_patterned_input(input_shape, pattern_size=(1, 1))

    # Initialize the PAM layer
    cam_layer = CAM(scale_gamma_initializer='ones')

    # Get the output from the PAM layer
    output_tensor = cam_layer(test_input)

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
    from tf_keras.models import Model
    from tf_keras.layers import Input

    # Define the input tensor
    # input_tensor = Input(shape=(64, 640, 1408))
    input_tensor = Input(shape=(None, None, 1408))

    # Create an instance of your custom layer
    attention_layer = CAM()

    # Apply your custom layer
    output_tensor = attention_layer(input_tensor)

    # Build the model
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # Summary of the model to see all parameters including those from PAM
    print(model.summary())
    
    print(f"\nPerforming unit testing:")
    ## 2. Unit testing
    ### 2.1 shape testing
    test_cam_shape((128, 128, 32))
    
    ### 2.2 Functionality testing
    test_cam_functionality((2, 20, 1408))
