import os
import ast
import inspect
import importlib.util
import sys

# Import trace_error for enhanced error handling
from nb_utils.error_handling import trace_error

def is_decorated_with_decorator(module, func_name, decorator_name):
    """
    Check if a function within a module has a specific decorator by parsing the AST.
    
    Args:
        module (module): The module where the function is located.
        func_name (str): The name of the function to check.
        decorator_name (str): The name of the decorator to look for.
    
    Returns:
        bool: True if the function is decorated with the specified decorator, False otherwise.
    
    Raises:
        If an unexpected error occurs, it logs a trace for deep inspection.
    """
    try:
        source = inspect.getsource(module)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                for decorator in node.decorator_list:
                    if ((isinstance(decorator, ast.Name) and decorator.id == decorator_name) or
                        (isinstance(decorator, ast.Attribute) and decorator.attr == decorator_name)):
                        return True
        return False
    except Exception as e:
        print(f"Error in AST parsing for {module.__name__}.{func_name}: {e}")
        trace_error()
        return False

def get_function_parameters(module, func_name):
    """
    Retrieve and format the parameters of a function, attempting to unwrap any decorators.
    
    Args:
        module (module): The module where the function is located.
        func_name (str): The name of the function from which to retrieve parameters.
    
    Returns:
        dict: A dictionary of parameter names and their default values, if any.
    
    Raises:
        Prints errors directly and attempts a fallback if the first retrieval fails.
    """
    try:
        func = getattr(module, func_name)
        # Attempt to unwrap the function if it's wrapped by decorators
        original_func = func
        while hasattr(original_func, '__wrapped__'):
            original_func = original_func.__wrapped__
        
        # Get the signature of the possibly unwrapped function
        signature = inspect.signature(original_func)
        params = {}
        for name, param in signature.parameters.items():
            if param.default is inspect.Parameter.empty:
                params[name] = 'no default'
            else:
                params[name] = param.default
        return params
    except Exception as e:
        print(f"Error retrieving parameters for {func.__name__}: {e}")
        trace_error()
        return {}

def find_decorated_functions(package_name, decorator_name):
    """
    Searches a package for all functions decorated with a specific decorator.
    
    Args:
        package_name (str): The name of the package to search.
        decorator_name (str): The name of the decorator to search for.
    
    Returns:
        list: A list of tuples containing the module name, function name, and parameters.
    
    Raises:
        Prints an error message if the package cannot be processed or if submodules can't be located.
    """
    decorated_functions = []
    package_spec = importlib.util.find_spec(package_name)
    if not package_spec or not package_spec.submodule_search_locations:
        print(f"No modules found for package {package_name}. Check if package is installed.")
        return decorated_functions

    package_path = package_spec.submodule_search_locations[0]

    for root, dirs, files in os.walk(package_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                module_path = os.path.join(root, file)
                module_name = module_path.replace(package_path, package_name).replace(os.sep, '.').replace('.py', '')
                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        if is_decorated_with_decorator(module, name, decorator_name):
                            params = get_function_parameters(module, name)
                            decorated_functions.append((module_name, name, params))
                except Exception as e:
                    print(f"Failed to import or process {module_name}: {e}")
                    trace_error()

    return decorated_functions

if __name__ == "__main__":
    # Example usage

    # set environment variable values before importing tensorflow and keras to use keras 2.15 in keras_cv_attention_models
    os.environ["KECAM_BACKEND"] = "tensorflow"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    # verify package import before proceeding further
    # import keras_cv_attention_models # direct import giving error in v1.4.1
    from keras_cv_attention_models import efficientformer

    package_name = 'keras_cv_attention_models'
    decorator_name = 'register_model'
    decorated_functions = find_decorated_functions(package_name, decorator_name)
    print(f"\nTotal {len(decorated_functions)} functions found decorated with @{decorator_name}\n")

    for module_name, func_name, params in decorated_functions:
        print(f"{module_name}.{func_name} with parameters: {params} \n")
