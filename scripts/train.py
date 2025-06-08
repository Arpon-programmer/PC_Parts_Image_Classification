import torch  # Import PyTorch for deep learning operations
from utils import utils_functions, get_tensorboard_writer, add_metrics_to_tensorboard  # Import utility functions and TensorBoard helpers
from model_architecture import SimpleCNN  # Import the CNN model architecture
from tqdm import tqdm  # Import tqdm for progress bars
from pc_parts_dataset import data_loader  # Import the data loader for the dataset

# =========================
# Data Loading and Model Initialization
# =========================
try:
    # Create the training data loader for the PC parts dataset
    train_loader = data_loader(
        data_path='/Users/arponbiswas/Computer-Vision-Projects/Image_classification_projects/PC_Parts_Image_Classification/Data/pc_parts_ready',  # Path to training data
        batch_size=32,  # Batch size for training
        train=True  # Indicate this is for training
    )

    # Instantiate the CNN model for image classification
    model = SimpleCNN()  # Create an instance of the model

    # Set up TensorBoard writer for logging training metrics and model graph
    writer = get_tensorboard_writer(
        log_dir='/Users/arponbiswas/Computer-Vision-Projects/Image_classification_projects/PC_Parts_Image_Classification/Tensorboard_Graph/Training_Graph/logs'  # Directory for TensorBoard logs
    )

    # Create a sample input tensor to visualize the model graph in TensorBoard
    sample_img = torch.rand(model.image_size)  # Generate a random tensor with the same shape as input images
    writer.add_graph(model, sample_img)  # Add model graph to TensorBoard

    # Set the learning rate for the optimizer
    lr = 0.001  # Learning rate for training

    # Select device: use GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose device

    # Move the model to the selected device
    model.to(device)  # Transfer model to GPU or CPU
except Exception as e:
    print(f"Error during initialization: {e}")  # Print any initialization errors

def train(model, train_loader, num_epochs=10, lr=0.001, is_scheduler=False, step_size=10, gamma=0.1):
    """
    Trains the given model using the provided data loader.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        is_scheduler (bool): Whether to use a learning rate scheduler.
        step_size (int): Step size for the scheduler (if used).
        gamma (float): Multiplicative factor of learning rate decay.

    Returns:
        None
    """
    try:
        # Initialize loss function, optimizer, and optionally scheduler
        if is_scheduler:
            loss_fn, optimizer, scheduler = utils_functions(
                model, lr, scheduler=is_scheduler, step_size=step_size, gamma=gamma  # Get loss, optimizer, scheduler if needed
            )
        else:
            loss_fn, optimizer = utils_functions(model, lr)  # Get loss and optimizer

        # Set the model to training mode
        model.train()  # Enable training-specific layers like dropout

        # Loop over the dataset for the specified number of epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0  # Accumulate loss for the epoch
            epoch_preds = []  # Store predictions for the epoch
            epoch_trues = []  # Store true labels for the epoch

            # Progress bar for monitoring training progress
            progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)  # Progress bar for batches

            # Iterate over each batch in the training loader
            for images, labels in progress_bar:
                # Move images and labels to the selected device
                images = images.to(device)  # Transfer images to device
                labels = labels.to(device)  # Transfer labels to device

                # Zero the parameter gradients before each batch update
                optimizer.zero_grad()  # Reset gradients

                # Forward pass: compute model outputs
                outputs = model(images)  # Get model predictions

                # Compute the loss between outputs and true labels
                loss = loss_fn(outputs, labels)  # Calculate loss

                # Backward pass: compute gradients
                loss.backward()  # Compute gradients

                # Update model parameters
                optimizer.step()  # Update weights

                # Accumulate batch loss
                epoch_loss += loss.item()  # Add batch loss to epoch loss

                # Store predictions and true labels for metrics
                epoch_preds.extend(outputs.argmax(dim=1).tolist())  # Store predicted labels
                epoch_trues.extend(labels.tolist())  # Store true labels

                # Update progress bar with current batch loss
                progress_bar.set_postfix(loss=loss.item())  # Show current loss

            # Compute average loss for the epoch
            epoch_loss = epoch_loss / len(train_loader)  # Average loss

            # Step the learning rate scheduler if enabled
            if is_scheduler:
                scheduler.step()  # Update learning rate

            # Log metrics (loss, accuracy, etc.) to TensorBoard
            add_metrics_to_tensorboard(writer, epoch_loss, epoch_trues, epoch_preds, epoch)  # Log metrics

            # Print epoch summary
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')  # Print epoch loss

        # Close the TensorBoard writer after training
        writer.close()  # Close writer
    except Exception as e:
        print(f"Error during training: {e}")  # Print training errors
        writer.close()  # Ensure writer is closed

if __name__ == "__main__":
    """
    Main entry point for training the model.
    Initializes training and handles exceptions.
    """
    try:
        # Start the training process with specified hyperparameters
        train(
            model,  # Model to train
            train_loader,  # DataLoader for training data
            num_epochs=10,  # Number of epochs
            lr=0.001,  # Learning rate
            is_scheduler=True,  # Use learning rate scheduler
            step_size=10,  # Scheduler step size
            gamma=0.1  # Scheduler gamma
        )
        print("Training completed successfully.")  # Indicate successful training
    except Exception as e:
        print(f"Error in main: {e}")  # Print errors from main