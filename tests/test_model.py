"""
test_model.py

This script defines and tests a simple image classification model for PC parts images.
It uses a sequential PyTorch model and a custom image processing function to predict
the class of each image in a specified test directory.

The script expects grayscale images of size 256x256 and predicts one of 14 PC part classes.
"""

import torch  # Import the PyTorch library
from test_data_preparation import process_images  # Import the process_images function from the test_data_preparation module

# model = torch.jit.load("model.pt")  # (Commented out) Load a pre-trained TorchScript model from file
model = torch.nn.Sequential(  # Define a simple sequential neural network model
    torch.nn.Flatten(),  # Flatten the input tensor
    torch.nn.Linear(256 * 256 * 1, 14)  # Linear layer mapping flattened input to 14 output classes
)

test_data_dir = "/Users/arponbiswas/Computer-Vision-Projects/Image_classification_projects/PC_Parts_Image_Classification/Data/test_images"  # Path to the directory containing test images
pc_parts_classes = [
    'CPU', 'GPU', 'Motherboard', 'RAM', 'SSD', 'HDD', 'Power Supply',
    'Cooling System', 'Case', 'Monitor', 'Keyboard', 'Mouse', 'Speaker', 'Webcam'
]  # List of class names for PC parts

def model_operation(test_data_dir, image_channel):
    """
    Processes images from the specified directory and predicts their classes using the model.

    Args:
        test_data_dir (str): Path to the directory containing test images.
        image_channel (int): Number of image channels (e.g., 1 for grayscale).

    Returns:
        tuple: (pred_images, image_names)
            pred_images (list): List of predicted class names for each image.
            image_names (list): List of image file names.
    """
    processed_images, image_names = process_images(test_data_dir, image_channel)  # Preprocess images and get their names
    pred_images = []  # Initialize a list to store predicted class names
    for processed_image in processed_images:  # Iterate over each processed image
        output = model(processed_image)  # Pass the image through the model to get output logits
        pred = torch.argmax(output, dim=1)  # Get the index of the class with the highest score
        pred_images.append(pc_parts_classes[pred])  # Map the predicted index to the class name and add to the list
    return pred_images, image_names  # Return the predicted class names and image names

if __name__ == "__main__":  # If this script is run directly
    outputs, image_names = model_operation(test_data_dir, 1)  # Call the model_operation function with test data and 1 channel (grayscale)
    for name, output in zip(image_names, outputs):  # Iterate over image names and their predicted classes
        print(f"Image: {name}, Predicted Class: {output}")  # Print the image name and its predicted class
