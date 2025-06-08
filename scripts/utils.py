import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def utils_functions(model, lr, scheduler=False, step_size=10, gamma=0.1):
    """
    Initializes and returns the loss function, optimizer, and optionally a learning rate scheduler for a PyTorch model.

    Args:
        model (torch.nn.Module): The neural network model whose parameters will be optimized.
        lr (float): Learning rate for the optimizer.
        scheduler (bool, optional): If True, returns a learning rate scheduler. Defaults to False.
        step_size (int, optional): Period of learning rate decay for the scheduler. Defaults to 10.
        gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.1.

    Returns:
        tuple: (loss_fn, optimizer) if scheduler is False,
               (loss_fn, optimizer, scheduler_obj) if scheduler is True.
               Returns None if an exception occurs.

    Example:
        loss_fn, optimizer = utils_functions(model, lr=0.001)
        loss_fn, optimizer, scheduler = utils_functions(model, lr=0.001, scheduler=True)
    """
    try:
        # Define the loss function for multi-class classification
        loss_fn = nn.CrossEntropyLoss()
        # Initialize the Adam optimizer with the model parameters and learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if scheduler:
            # Optionally, create a StepLR scheduler for learning rate decay
            scheduler_obj = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            return loss_fn, optimizer, scheduler_obj
        else:
            # Return only loss function and optimizer if scheduler is not requested
            return loss_fn, optimizer
    except Exception as e:
        # Print error message and return None if any exception occurs
        print(f"Error in utils_functions: {e}")
        return None

def acc_f1_pre_rec(y_true, y_pred):
    """
    Calculates evaluation metrics: accuracy, F1 score, precision, recall, and confusion matrix.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        tuple: (accuracy, f1_score, precision, recall, confusion_matrix)
               Returns zeros and None for confusion matrix if an exception occurs.

    Example:
        acc, f1, pre, rec, cm = acc_f1_pre_rec(y_true, y_pred)
    """
    try:
        # Compute accuracy
        acc = accuracy_score(y_true, y_pred)
        # Compute weighted F1 score
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        # Compute weighted precision
        pre = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        # Compute weighted recall
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Return all computed metrics
        return acc, f1, pre, rec, cm
    except Exception as e:
        # Print error message and return zeros and None if any exception occurs
        print(f"Error in acc_f1_pre_rec: {e}")
        return 0, 0, 0, 0, None

def get_tensorboard_writer(log_dir):
    """
    Creates a TensorBoard SummaryWriter for logging metrics and visualizations.

    Args:
        log_dir (str): Directory where TensorBoard logs will be saved.

    Returns:
        SummaryWriter: TensorBoard writer object for logging.
                       Returns None if an exception occurs.

    Example:
        writer = get_tensorboard_writer('./logs')
    """
    try:
        # Initialize the TensorBoard SummaryWriter with the given log directory
        writer = SummaryWriter(log_dir=log_dir)
        # Return the writer object
        return writer
    except Exception as e:
        # Print error message and return None if any exception occurs
        print(f"Error in get_tensorboard_writer: {e}")
        return None

def add_metrics_to_tensorboard(writer, loss, y_true, y_pred, epoch):
    """
    Logs loss, accuracy, F1 score, precision, recall, and confusion matrix to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer object.
        loss (float): Loss value to log.
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        epoch (int): Current epoch number (used as the global step in TensorBoard).

    Returns:
        None

    Example:
        add_metrics_to_tensorboard(writer, loss, y_true, y_pred, epoch)
    """
    try:
        # Calculate metrics using the helper function
        acc, f1, pre, rec, cm = acc_f1_pre_rec(y_true, y_pred)
        # Log scalar metrics to TensorBoard
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('Accuracy', acc, epoch)
        writer.add_scalar('F1 Score', f1, epoch)
        writer.add_scalar('Precision', pre, epoch)
        writer.add_scalar('Recall', rec, epoch)
        # If confusion matrix is available, log it as a heatmap figure
        if cm is not None:
            # Create a new matplotlib figure and axis
            fig, ax = plt.subplots()
            # Plot the confusion matrix as a heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            # Set axis labels and title
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            # Add the figure to TensorBoard
            writer.add_figure('Confusion Matrix', fig, global_step=epoch)
            # Close the figure to free memory
            plt.close(fig)
    except Exception as e:
        # Print error message if any exception occurs
        print(f"Error in add_metrics_to_tensorboard: {e}")