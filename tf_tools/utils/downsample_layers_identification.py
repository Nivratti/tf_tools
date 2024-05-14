
def generate_target_shapes(input_height, input_width):
    """
    Generate a list of target shapes by iteratively halving the dimensions of the input,
    until both dimensions are reduced to a value of 2 or less.

    Args:
    input_height (int): The initial height of the input.
    input_width (int): The initial width of the input.

    Returns:
    list of tuple: A list of tuples, where each tuple contains the halved dimensions 
                   (height, width) from the input dimensions, stopping when both are 
                   2 or smaller.
    """
    target_shapes = []
    while input_height > 2 and input_width > 2:
        input_height //= 2
        input_width //= 2
        target_shapes.append((input_height, input_width))
    return target_shapes

def find_downsample_layers(model, input_height, input_width, verbose=False):
    """
    Identify and return the last layer in each target shape category that matches the 
    dimensions obtained from the `generate_target_shapes` function, from a predefined 
    model's layers.

    Args:
    input_height (int): The initial height of the input.
    input_width (int): The initial width of the input.
    verbose (bool, optional): If True, print detailed information about target shapes 
                              and matching layers. Defaults to False.

    Returns:
    dict: A dictionary where each key is a tuple representing the target shape, and the 
          value is the last layer's name and shape that matches this target shape.
    """
    target_shapes = generate_target_shapes(input_height, input_width)
    if verbose:
        print(f"target_shapes: {target_shapes}")

    downsamples = {target_shape: [] for target_shape in target_shapes}

    for layer in model.layers:
        if hasattr(layer.output, 'shape'):
            shape = layer.output.shape
            for target_shape in target_shapes:
                if tuple(shape[1:3]) == target_shape:
                    downsamples[target_shape].append((layer.name, shape))
                    break

    if verbose:
        print(f"All matching layers:")
        for target_shape, layers in downsamples.items():
            print(f"Target Shape {target_shape}:")
            for layer_name, shape in layers:
                print(f"  Layer: {layer_name}, Output Shape: {shape}")

    last_layers = {target_shape: layers[-1] for target_shape, layers in downsamples.items() if layers}

    if verbose:
        for target_shape, (layer_name, shape) in last_layers.items():
            print(f"Target Shape {target_shape}: Layer {layer_name}")

    return last_layers