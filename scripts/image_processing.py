from torchvision import transforms  # Import torchvision transforms module

# Define the transformation pipeline for training images
train_trans = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256 pixels
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip image horizontally with 50% probability
    transforms.RandomRotation(10),  # Randomly rotate image by up to 10 degrees
    transforms.RandomAffine(
        degrees=0,  # No additional rotation
        translate=(0.05, 0.05),  # Randomly translate image by up to 5% horizontally and vertically
        scale=(0.9, 1.1)  # Randomly scale image between 90% and 110%
    ),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3, interpolation=3),  # Apply random perspective transformation with 30% probability
    transforms.ColorJitter(
        brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05  # Randomly change brightness, contrast, saturation, and hue
    ),
    transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
    transforms.Normalize(
        mean=[0.5597623586654663, 0.5584744215011597, 0.5615870952606201],  # Normalize tensor with mean
        std=[0.2830338776111603, 0.27588963508605957, 0.2759666442871094]    # and standard deviation
    )
])

# Define the transformation pipeline for validation images
val_trans = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256 pixels
    transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
    transforms.Normalize(
        mean=[0.5597623586654663, 0.5584744215011597, 0.5615870952606201],  # Normalize tensor with mean
        std=[0.2830338776111603, 0.27588963508605957, 0.2759666442871094]    # and standard deviation
    )
])

def train_transform(image):
    """
    Applies a predefined set of transformations to an input image for training purposes.

    This function attempts to apply the `train_trans` transformation pipeline to the provided image.
    If an error occurs during the transformation process, it catches the exception, prints an error message,
    and returns None to indicate failure.

    Parameters:
        image (PIL.Image.Image or numpy.ndarray): The input image to be transformed. The image should be in a format
            compatible with the `train_trans` transformation pipeline.

    Returns:
        Transformed image object: The result of applying `train_trans` to the input image, typically a tensor or
            processed image ready for model input.
        None: If an exception occurs during transformation, None is returned.

    Exceptions:
        Any exception raised during the transformation process is caught and logged to the console with a descriptive
        error message.

    Example:
        >>> transformed_img = train_transform(img)
        >>> if transformed_img is None:
        ...     print("Transformation failed.")
    """
    try:
        return train_trans(image)  # Apply training transformations to the input image
    except Exception as e:
        print(f"Error in train_transform: {e}")  # Print error message if transformation fails
        return None  # Return None if exception occurs

def val_transform(image):
    try:
        return val_trans(image)  # Apply validation transformations to the input image
    except Exception as e:
        print(f"Error in val_transform: {e}")  # Print error message if transformation fails
        return None  # Return None if exception occurs