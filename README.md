# OCR for Character Recognition with CNNs

This project utilizes convolutional neural networks (CNNs) to recognize individual characters from images, achieving 97.2% accuracy on the provided dataset using PyTorch for model building and training.

## Instructions

- **Organize Dataset:** Arrange the dataset in the specified folder structure, with paths for training and testing data.
- **Rebuild Dataset:** Set the `REBUILD_DATA` flag to `True` during the initial run to build the dataset.
- **Train the Model:** Execute the training loop for the desired number of epochs.
- **Evaluate Accuracy:** Assess the model on the test dataset to obtain accuracy metrics.

## Important Notes

- **Dataset Organization:** Ensure the dataset folders are properly organized and paths are correctly set.
- **Hyperparameter Tuning:** Adjust hyperparameters such as batch size, learning rate, and number of epochs to enhance performance.
- **Customization:** Modify the model architecture or training process according to specific needs and constraints.

## Project Structure

### Data Preparation

The dataset should be organized into folders representing each class (e.g., letters A-Z, digits 0-9). The code reads images from the specified training and testing paths, preprocesses them (resizing to a fixed size), and constructs the training and testing datasets.

1. **Folder Organization:** Each folder corresponds to a class (e.g., A-Z, 0-9).
2. **Image Preprocessing:** Images are resized to a fixed size to ensure consistency.
3. **Dataset Construction:** Separate datasets are created for training and testing.

### Model Design

The neural network consists of:

1. **Convolutional Layers:** Three convolutional layers, each followed by a max-pooling layer and a ReLU activation function.
2. **Fully Connected Layers:** Two linear layers following the convolutional layers.
3. **Output Layer:** Contains 36 units, corresponding to the 36 classes (26 letters and 10 digits).

### Training Process

The model is trained using the Adam optimizer and a mean squared error (MSE) loss function. The training involves:

1. **Batch Processing:** Data is processed in batches to optimize training.
2. **Multiple Epochs:** Training occurs over several epochs to improve accuracy.
3. **Progress Monitoring:** Training progress and loss are printed for each epoch.

### Model Evaluation

After training, the model is evaluated on the test dataset to measure its accuracy. The evaluation includes:

1. **Prediction Comparison:** Comparing predicted classes with true classes.
2. **Accuracy Calculation:** Determining the ratio of correct predictions to the total number of samples to measure accuracy.

## Example Workflow

1. **Set Up Environment:** Ensure PyTorch and other dependencies are installed.
2. **Prepare Data:** Organize your dataset into the required folder structure.
3. **Configure Settings:** Set `REBUILD_DATA` to `True` for the initial run, and adjust hyperparameters as needed.
4. **Train Model:** Run the training loop for the specified number of epochs.
5. **Evaluate Model:** Test the model on the test dataset and review the accuracy metrics.

## Customization Tips

- **Modify Architecture:** Adjust the number of layers or units in each layer to experiment with different network architectures.
- **Optimize Hyperparameters:** Tune the learning rate, batch size, and number of epochs to find the best configuration for your data.
- **Data Augmentation:** Implement data augmentation techniques to improve the robustness of the model.

By following these guidelines and utilizing the provided code, you can effectively recognize individual characters from images using CNNs. Adjust and experiment with different configurations to achieve the best results for your specific dataset.
