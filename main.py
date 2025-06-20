import torch  # PyTorch for model loading and tensor operations
import torchvision.transforms as transforms  # For image preprocessing
from PIL import Image  # For image loading and manipulation

# List of class names for PC parts classification
pc_parts_classes = [
    'cables', 'case', 'cpu', 'gpu', 'hdd', 'headset', 'keyboard',
    'microphone', 'monitor', 'motherboard', 'mouse', 'ram', 'speakers', 'webcam'
]

# Load the pre-trained model with error handling
try:
    model = torch.jit.load('Path of the model', map_location='cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image_path, img_type="RGB"):
    """
    Preprocesses an image for model inference.

    Args:
        image_path (str): Path to the input image.
        img_type (str): Image type, either 'RGB' or 'grayscale'.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, C, H, W).
    """
    try:
        # Open the image and convert to the specified type
        if img_type.lower() == "rgb":
            image = Image.open(image_path).convert("RGB")
        elif img_type.lower() == "grayscale":
            image = Image.open(image_path).convert("L")
        else:
            raise ValueError("Unsupported image type. Use 'RGB' or 'grayscale'.")
        # Define the preprocessing transformations: resize and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize image to 256x256
            transforms.ToTensor()           # Convert image to PyTorch tensor
        ])
        # Apply transformations and add batch dimension
        return transform(image).unsqueeze(0)
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def operation(img_path):
    """
    Runs the model on the given image and returns the predicted class name.

    Args:
        img_path (str): Path to the input image.

    Returns:
        str: Predicted class name.
    """
    if model is None:
        return "Model not loaded."
    try:
        processed_img = preprocess_image(img_path)  # Preprocess the image
        out = model(processed_img)                  # Run the model to get predictions
        pred = torch.argmax(out, dim=1)             # Get the index of the highest score
        pred_img = pc_parts_classes[pred]           # Map index to class name
        return pred_img
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    # Main loop for user input and prediction
    while True:
        try:
            img_path = input('Give the image path : ')  # Prompt user for image path
            prediction = operation(img_path)            # Get prediction
            print(f"Predicted image {prediction}.")     # Print predicted class
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")