"""
Module: test_data_preparation

This module provides utilities for processing images in a specified directory for image classification tasks.
It applies a sequence of transformations (resize and tensor conversion) to each image and supports both grayscale
and RGB images based on the specified number of channels.

Functions:
    process_images(test_data_dir, image_channel):
        Processes all images in the given directory, applies transformations, and returns a list of processed
        image tensors along with their corresponding names (without extensions).

Example usage:
    processed_images, image_names = process_images('/path/to/images', image_channel=3)
"""

import torchvision.transforms as transforms  # Import image transformation utilities from torchvision
from PIL import Image  # Import the Python Imaging Library for image processing
import os  # Import os module for interacting with the operating system

# Define a sequence of image transformations: resize to 256x256 and convert to tensor
trans = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256 pixels
    transforms.ToTensor()           # Convert image to PyTorch tensor
    ])

def process_images(test_data_dir, image_channel):
    """
    Processes images in the specified directory by resizing and converting them to tensors.

    Args:
        test_data_dir (str): Path to the directory containing image files.
        image_channel (int): Number of image channels (1 for grayscale, 3 for RGB).

    Returns:
        tuple: (processed_images, image_names)
            processed_images (list): List of processed image tensors.
            image_names (list): List of image file names without extensions.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Notes:
        Only files with extensions .jpg, .jpeg, or .png are processed.
        The function applies resizing to 256x256 pixels and converts images to tensors.
    """
    processed_images = []  # List to store processed image tensors
    image_names = []       # List to store image names (without extension)
    if not os.path.exists(test_data_dir):  # Check if the directory exists
        raise FileNotFoundError(f"The directory {test_data_dir} does not exist.")  # Raise error if not found
    for filename in os.listdir(test_data_dir):  # Iterate over each file in the directory
        image_names.append(filename.split(".")[0])  # Add the filename (without extension) to image_names
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):  # Check for image files
            image_path = os.path.join(test_data_dir, filename)  # Get the full path of the image
            if image_channel == 1:
                image = Image.open(image_path).convert('L')  # Open image and convert to grayscale if channel is 1
            if image_channel == 3:
                image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB if channel is 3
            processed_image = trans(image).unsqueeze(0)  # Apply transformations and add batch dimension
            processed_images.append(processed_image)  # Add processed image tensor to the list
    return processed_images, image_names  # Return the list of processed images and their names