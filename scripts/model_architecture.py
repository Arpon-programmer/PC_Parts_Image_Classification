import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    SimpleCNN(nn.Module)

    A simple Convolutional Neural Network (CNN) architecture for image classification tasks.

    Attributes:
        version (str): Version identifier for the model architecture.
        image_size (tuple): Expected input image size in the format (batch_size, channels, height, width).
        layer1 (nn.Sequential): Sequential container for the first convolutional block, consisting of:
            - nn.Conv2d: 2D convolutional layer with 1 input channel, 1 output channel, 6x6 kernel, stride 4, no padding.
            - nn.ReLU: Rectified Linear Unit activation function.
            - nn.MaxPool2d: Max pooling layer with 3x3 kernel and stride 3.
        flatten (nn.Flatten): Layer to flatten the output from the convolutional block.
        fc1 (nn.Linear): Fully connected linear layer mapping the flattened features to 14 output classes.

    Methods:
        __init__():
            Initializes the layers of the SimpleCNN model. Handles exceptions during layer initialization and prints errors if any occur.

        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 1, 256, 256).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, 14) representing class scores.
            Handles exceptions during the forward pass and prints errors if any occur.

    Usage:
        model = SimpleCNN()
        output = model(input_tensor)

    Note:
        - The model expects grayscale images (1 channel) of size 256x256.
        - The architecture is minimal and intended for demonstration or simple classification tasks.
        - Exception handling is included for both initialization and forward pass for debugging purposes.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()  # Call the parent class (nn.Module) constructor
        self.version = '1.0'  # Model version identifier
        self.image_size = (1, 1, 256, 256)  # Expected input image size (batch, channel, height, width)
        try:
            # Define the first convolutional block
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=6, stride=4, padding=0),  # 2D convolution layer: 1 input channel, 1 output channel, 6x6 kernel, stride 4, no padding
                nn.ReLU(),  # Activation function: ReLU introduces non-linearity
                nn.MaxPool2d(kernel_size=3, stride=3),  # Max pooling: reduces spatial dimensions with 3x3 kernel and stride 3
            )
            self.flatten = nn.Flatten()  # Flatten the output from conv block to a 1D tensor for the fully connected layer
            self.fc1 = nn.Linear(1 * 21 * 21, 14)  # Fully connected layer: maps flattened features to 14 output classes
        except Exception as e:
            print(f"Error initializing layers: {e}")  # Print error if layer initialization fails

    def forward(self, x):
        try:
            out = self.layer1(x)  # Pass input through the convolutional block
            out = self.flatten(out)  # Flatten the output to feed into the fully connected layer
            out = self.fc1(out)  # Pass through the fully connected layer to get class scores
            return out  # Return the output tensor (class scores)
        except Exception as e:
            print(f"Error in forward pass: {e}")  # Print error if forward pass fails