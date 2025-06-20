from PIL import Image, UnidentifiedImageError
import os

import torchvision.transforms as transforms

trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
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
    processed_images = []
    image_names = []
    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"The directory {test_data_dir} does not exist.")
    for filename in os.listdir(test_data_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_names.append(filename.split(".")[0])
            image_path = os.path.join(test_data_dir, filename)
            try:
                if image_channel == 1:
                    image = Image.open(image_path).convert('L')
                elif image_channel == 3:
                    image = Image.open(image_path).convert('RGB')
                else:
                    raise ValueError("image_channel must be 1 (grayscale) or 3 (RGB)")
                processed_image = trans(image).unsqueeze(0)
                processed_images.append(processed_image)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Warning: Could not process image {filename}: {e}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            # Optionally, skip non-image files silently or log them
            continue
    return processed_images, image_names