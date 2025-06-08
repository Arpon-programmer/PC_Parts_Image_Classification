# PC Parts Image Classification

A deep learning project for classifying images of PC components (e.g., CPU, GPU, RAM, etc.) using Convolutional Neural Networks (CNNs) in PyTorch. The project features modular code, TensorBoard logging, and tools for data exploration and model analysis.

## Features
- Modular codebase with clear separation of data, models, scripts, and notebooks
- Data exploration and visualization tools
- Custom CNN architecture for image classification
- Training, evaluation, and inference scripts
- TensorBoard integration for metrics and feature map visualization
- Configurable via YAML files
- Test scripts for data loading and model validation

## Project Structure
```
├── Data/                  # Image dataset organized by class
├── Model_scripts/         # Model scripts
├── models/                # Saved models/checkpoints
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── scripts/               # Training, evaluation, and utility scripts
├── Tensorboard_Graph/     # TensorBoard logs
├── tests/                 # Test scripts
├── config/                # YAML config files
├── requirements.txt       # Python dependencies
├── LICENSE                # Project license (MIT)
├── README.md              # Project documentation
```

## Installation
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd PC_Parts_Image_Classification
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
- **Data Exploration:**
  - Use `notebooks/data_exploration.ipynb` to visualize and inspect your dataset.
- **Training:**
  - Run `python scripts/train.py` to train the model. Training logs and model graphs will be saved to `Tensorboard_Graph/`.
- **Evaluation:**
  - Use `scripts/evaluate.py` or the provided notebooks to evaluate model performance.
- **TensorBoard:**
  - Launch TensorBoard to visualize training progress:
    ```sh
    tensorboard --logdir Tensorboard_Graph/
    ```

## Configuration
- Edit YAML files in the `config/` directory to adjust model, training, and evaluation parameters.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
