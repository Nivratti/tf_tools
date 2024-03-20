import os
import re
from nb_utils.file_dir_handling import list_files

def collect_tfrecord_files(input_paths):
    """
    Collects TFRecord file paths from a list of input paths.

    This function iterates over the provided input paths. If an input path is a file, it is assumed to be a TFRecord file
    and is added directly to the list. If an input path is a directory, the function searches the directory for files
    with a '.tfrec' extension and adds them to the list.

    Args:
    input_paths (list of str): List of input paths. Each path can be a directory or a direct file path.

    Returns:
    list of str: A list of TFRecord file paths.

    Note:
    - The function assumes that files with a '.tfrec' extension are TFRecord files.
    - The function `list_files`, used to list files within directories, should be defined to filter TFRecord files.

    # Example usage:
    # training_tfrecord_paths = ["path/to/directory", "path/to/file.tfrec"]
    # training_tfrecord_filepaths = collect_tfrecord_paths(training_tfrecord_paths)
    """
    tfrecord_filepaths = []
    for path in input_paths:
        if os.path.isfile(path):
            tfrecord_filepaths.append(path)  # Directly add the file path
        else:
            # List and add TFRecord files from the directory
            tfrecord_files = list_files(path, filter_ext=[".tfrec"])
            tfrecord_filepaths.extend(tfrecord_files)
    return tfrecord_filepaths

def calculate_total_image_count(tfrecord_files):
    """
    Calculates the total count of images from a list of TFRecord filenames.

    Args:
    tfrecord_files (list of str): List of TFRecord file paths.

    Returns:
    int: Total count of images across all TFRecord files.

    # Example usage:
    >> tfrecord_files = list_files(tfrecords_dir, filter_ext=[".tfrec"])
    >> total_images = calculate_total_image_count(tfrecord_files)
    """
    total_count = 0
    for file in tfrecord_files:
        # Extract the number following 'count-' or the last number in the filename
        match = re.search(r"count-(\d+).tfrec$", os.path.basename(file))
        if match:
            count = int(match.group(1))
            total_count += count
    return total_count