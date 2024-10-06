import os
import sys
import re
from nb_utils.file_dir_handling import list_files

def list_tfrecord_files_in_gcs_directory(gcs_url):
    """
    Lists all .tfrec files in a specified GCS directory URL and returns their full URLs.

    Args:
        gcs_url (str): The URL of the GCS directory from which to list .tfrec files.
                       Expected format: 'gs://bucket-name/path/to/directory'

    Returns:
        list[str]: A list of URLs pointing to the .tfrec files in the specified GCS directory.
    """
    from google.cloud import storage
    
    # Split the GCS URL into bucket name and blob prefix, assuming the URL starts with 'gs://'
    if "gs://" in gcs_url:
        gcs_url = gcs_url.split("gs://")[1]
    bucket_name, prefix = gcs_url.split("/", 1)

    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket object from the bucket name
    bucket = client.bucket(bucket_name)

    # List all blobs in the specified directory and filter by the suffix '.tfrecord'
    blobs = client.list_blobs(bucket, prefix=prefix)

    # Initialize a list to hold the full URLs of the .tfrecord files
    file_urls = []
    for blob in blobs:
        if blob.name.endswith('.tfrec'):  # Corrected the file extension check
            # Construct the full URL for each .tfrec file
            file_url = os.path.join("gs://", bucket.name, blob.name)
            file_urls.append(file_url)

    return file_urls

def collect_tfrecord_files(paths):
    """
    Collects TFRecord file paths from a given list of paths. Each path can be a directory, a file, or a GCS URL.

    Parameters
    ----------
    paths : list of str
        A list where each element can be a directory path, a direct file path, or a Google Cloud Storage (GCS) URL.

    Returns
    -------
    list of str
        A compiled list of paths to TFRecord files found based on the input criteria.

    Raises
    ------
    FileNotFoundError
        If any of the provided local paths do not exist.
    ValueError
        If a path is not a file, directory, or a valid GCS URL.

    Notes
    -----
    - The function assumes that files ending with '.tfrec' are TFRecord files.
    - The `list_files` function used to list files within directories should be capable of filtering by extension, specifically for '.tfrec'.
    - The existence of GCS paths is assumed to be validated within the `list_tfrecord_files_in_gcs_directory` function.

    Example
    -------
    >>> example_paths = ["path/to/directory", "path/to/file.tfrec", "gs://bucket/path/"]
    >>> collected_tfrecord_paths = collect_tfrecord_files(example_paths)
    """
    collected_file_paths = []
    for path in paths:
        if path.startswith("gs://"):
            # Collect TFRecord files from a GCS directory
            try:
                tfrecord_files = list_tfrecord_files_in_gcs_directory(path)
                collected_file_paths.extend(tfrecord_files)
            except Exception as e:
                print(f"Error accessing GCS path '{path}': {e}", file=sys.stderr)
                sys.exit(1)
        elif os.path.isfile(path):
            if path.endswith('.tfrec'):
                # Directly add the file path if it ends with '.tfrec'
                collected_file_paths.append(path)
            else:
                print(f"File '{path}' does not have a '.tfrec' extension.", file=sys.stderr)
                sys.exit(1)
        elif os.path.isdir(path):
            # List and add TFRecord files from the directory
            try:
                tfrecord_files = list_files(path, filter_ext=[".tfrec"])
                collected_file_paths.extend(tfrecord_files)
            except Exception as e:
                print(f"Error accessing directory '{path}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Path does not exist or is not a file/directory/GCS URL
            print(f"Path '{path}' does not exist or is not a valid file/directory/GCS URL.", file=sys.stderr)
            sys.exit(1)
    return collected_file_paths

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

def verify_tfrecord(file_path):
    """
    Verifies the integrity of a TFRecord file.

    This function attempts to iterate through all records in the specified
    TFRecord file to check for any corruption or errors. If the file is valid,
    it prints a confirmation message. If an error is encountered, it catches
    the exception and prints an error message.

    Args:
        file_path (str): The path to the TFRecord file to be verified.

    Raises:
        Exception: Catches any exception raised during the verification process
                   and prints an error message with the exception details.
    Usage:
        for file in tfrecord_filepaths:
            verify_tfrecord(file)
    """
    try:
        for record in tf.data.TFRecordDataset(file_path, compression_type="GZIP"):
            pass
        print(f"{file_path} is valid.")
    except Exception as e:
        print(f"Error in {file_path}: {e}")
