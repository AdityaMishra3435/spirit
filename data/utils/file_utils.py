"""
Utility functions for file operations.
"""

import os
import re
import glob
import shutil
import datetime
from pathlib import Path


def ensure_dir(directory):
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory (str or Path): Directory path to ensure exists.
    """
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True)
        print(f"Created directory: {directory}")


def copy_file(source, destination):
    """
    Copy a file from source to destination.

    Args:
        source (str or Path): Source file path.
        destination (str or Path): Destination file path.
    """
    try:
        shutil.copy(source, destination)
        return True
    except Exception as e:
        print(f"Error copying {source} to {destination}: {e}")
        return False


def get_files_by_pattern(directory, pattern):
    """
    Get list of files matching pattern in a directory.

    Args:
        directory (str or Path): Directory to search in.
        pattern (str): Glob pattern to match files.

    Returns:
        list: List of matched file paths.
    """
    return glob.glob(os.path.join(directory, pattern))


def parse_and_format_timestamp(timestamp, input_format, output_format):
    """
    Parse timestamp and format it to different format.

    Args:
        timestamp (str): Timestamp string to parse.
        input_format (str): Format of input timestamp.
        output_format (str): Desired output format.

    Returns:
        str: Formatted timestamp.
    """
    date_time = datetime.datetime.strptime(timestamp, input_format)
    return date_time.strftime(output_format)
