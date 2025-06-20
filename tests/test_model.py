# Import the torch library for building and running neural networks
import torch
# Import the process_images function from the test_data_preparation module for image preprocessing
from test_data_preparation import process_images

# Define a simple neural network model using PyTorch's Sequential API
# The model flattens the input and applies a linear layer to classify into 14 classes
model = torch.nn.Sequential(
    torch.nn.Flatten(),  # Flattens the input tensor (e.g., from [batch, 1, 256, 256] to [batch, 65536])
    torch.nn.Linear(256 * 256 * 1, 14)  # Linear layer mapping flattened input to 14 output classes
)

# Path to the directory containing test images
test_data_dir = "/Users/arponbiswas/Computer-Vision-Projects/Image_classification_projects/PC_Parts_Image_Classification/Data/test_images"
# List of class names corresponding to PC parts
pc_parts_classes = [
    'cables', 'case', 'cpu', 'gpu', 'hdd', 'headset', 'keyboard',
    'microphone', 'monitor', 'motherboard', 'mouse', 'ram', 'speakers', 'webcam'
]

def model_operation(test_data_dir, image_channel):
    """
    Processes test images and predicts their classes using the defined model.

    Args:
        test_data_dir (str): Directory containing test images.
        image_channel (int): Number of image channels (e.g., 1 for grayscale).

    Returns:
        tuple: (predicted class names, image file names)
    """
    try:
        # Preprocess images and get their file names
        processed_images, image_names = process_images(test_data_dir, image_channel)
    except Exception as e:
        # Print error if image processing fails and return empty lists
        print(f"Error processing images: {e}")
        return [], []
    pred_images = []  # List to store predicted class names
    for idx, processed_image in enumerate(processed_images):
        try:
            # Pass the processed image through the model to get output logits
            output = model(processed_image)
            # Get the index of the class with the highest score
            pred = torch.argmax(output, dim=1)
            # Map the predicted index to the class name and append to results
            pred_images.append(pc_parts_classes[pred])
        except Exception as e:
            # Print error if prediction fails for an image and append "Error"
            print(f"Error predicting image '{image_names[idx]}': {e}")
            pred_images.append("Error")
    # Return the list of predicted class names and corresponding image names
    return pred_images, image_names

if __name__ == "__main__":
    # Main execution block for running predictions on test images
    try:
        # Get predicted outputs and image names
        outputs, image_names = model_operation(test_data_dir, 1)
        # Print the predicted class for each image
        for name, output in zip(image_names, outputs):
            print(f"Image: {name}, Predicted Class: {output}")
    except Exception as e:
        # Print any unexpected errors that occur during execution
        print(f"Unexpected error: {e}")
