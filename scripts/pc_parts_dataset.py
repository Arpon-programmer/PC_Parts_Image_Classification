from torch.utils.data import DataLoader  # Import DataLoader for batching data
from torchvision.datasets import ImageFolder  # Import ImageFolder for loading images from directories
from image_processing import train_transform, val_transform  # Import image transformations

def data_loader(data_path, batch_size=32, train=True):
    """
    Loads image data from a specified directory and returns a DataLoader for batch processing.

    This function utilizes the PyTorch `ImageFolder` dataset class to load images from the given
    directory (`data_path`). It applies different transformations depending on whether the data
    is intended for training or validation/testing. The images are then wrapped in a DataLoader
    to enable efficient mini-batch loading and optional shuffling.

    Args:
        data_path (str): 
            The root directory path containing the image data, organized in subdirectories by class.
        batch_size (int, optional): 
            Number of samples per batch to load. Defaults to 32.
        train (bool, optional): 
            If True, applies training transformations and enables shuffling. 
            If False, applies validation/test transformations and disables shuffling. Defaults to True.

    Returns:
        DataLoader or None: 
            Returns a PyTorch DataLoader object for the dataset if loading is successful.
            Returns None if an exception occurs during loading.

    Raises:
        None: 
            All exceptions are caught internally and an error message is printed.

    Notes:
        - The function expects `train_transform` and `val_transform` to be defined in the scope.
        - The directory structure under `data_path` should be compatible with PyTorch's `ImageFolder`:
          each class should have its own subdirectory containing images.
        - If an error occurs (e.g., missing directory, invalid transforms), the function prints an error
          message and returns None.
    """
    try:
        # Create an ImageFolder dataset with the appropriate transform
        dataset = ImageFolder(
            root=data_path,  # Root directory containing images organized by class
            transform=train_transform if train else val_transform  # Use train or validation transform
        )
        # Create a DataLoader for batching and (optionally) shuffling the data
        loader = DataLoader(
            dataset,  # The dataset to load from
            batch_size=batch_size,  # Number of samples per batch
            shuffle=train  # Shuffle if training, else no shuffle
        )
        return loader  # Return the DataLoader object
    except Exception as e:
        # Print error message if loading fails and return None
        print(f"Error loading data: {e}")
        return None