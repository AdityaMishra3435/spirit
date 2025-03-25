"""
Module for downloading images from the NREL website.
"""

import os
import requests
import datetime
import io
import zipfile
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
import sys
sys.path.append("..")
from utils.file_utils import ensure_dir


def extract_zip_url(url):
    """
    Fetches the page content from the given URL and extracts the ZIP file URL.

    Args:
        url (str): The URL to fetch HTML content from.

    Returns:
        str: The URL of the ZIP file or None if not found.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Find link that contains zip in href
        link = soup.find("a", href=re.compile(r".*\.zip"))
        if link:
            return link.get("href")
    return None


def download_zip_and_extract(zip_url, save_dir):
    """
    Downloads the zip file from the given URL and extracts it into the specified save directory.

    Args:
        zip_url (str): The URL of the zip file.
        save_dir (str): The directory to save extracted files.

    Returns:
        bool: True if successful, False otherwise
    """
    ensure_dir(save_dir)

    response = requests.get(zip_url)
    if response.status_code == 200:
        # Check if the content is a valid ZIP file
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(save_dir)
            return True
        except zipfile.BadZipFile:
            print(f"Error: The downloaded file from {zip_url} is not a valid ZIP file.")
            with open(os.path.join(save_dir, "invalid_file.html"), "wb") as f:
                f.write(response.content)
            return False
    else:
        print(
            f"Failed to download the file from {zip_url}. Status code: {response.status_code}"
        )
        return False


def process_zip_download(url, save_dir):
    """
    Processes the given URL to download and extract the ZIP file of images.

    Args:
        url (str): The URL to fetch HTML content from.
        save_dir (str): The directory to save the extracted images.

    Returns:
        bool: True if successful, False if no zip URL found
    """
    # Extract the zip URL
    zip_url = extract_zip_url(url)
    if not zip_url:
        print(f"No zip URL found at {url}")
        return False

    # Make sure the ZIP URL is absolute
    zip_url = urljoin(url, zip_url)

    return download_zip_and_extract(zip_url, save_dir)


def download_images_in_range(
    start_date,
    end_date,
    base_save_path,
    base_url="https://midcdmz.nrel.gov/apps/imageranim.pl",
):
    """
    Downloads and extracts images in the given date range into a user-specified directory.

    Args:
        start_date (datetime.date): The start date for downloading images.
        end_date (datetime.date): The end date for downloading images.
        base_save_path (str): The base directory where the zip files and images will be saved.
        base_url (str): Base URL for the image service.

    Returns:
        int: Number of days successfully processed
    """
    current_date = start_date
    successful_days = 0

    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day

        # Construct the URL for the current date
        url = f"{base_url}?site=SRRLASI;year={year};month={month};day={day};type="

        # Create a specific directory for the current date inside the base path
        save_dir = os.path.join(base_save_path, f"{year}-{month:02d}-{day:02d}")

        print(f"Processing data for {current_date} and saving to {save_dir}")

        # Process the zip download
        success = process_zip_download(url, save_dir)
        if success:
            successful_days += 1

        # Move to the next day
        current_date += datetime.timedelta(days=1)

    print(
        f"Download complete. Successfully processed {successful_days} out of {(end_date - start_date).days + 1} days."
    )
    return successful_days
