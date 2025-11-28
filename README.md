# Multilayer Perceptron

## Introduction to Neural Networks: Theory and Implementation

This project implements a multilayer perceptron (MLP) from scratch for breast cancer classification using the Wisconsin Breast Cancer dataset.

## Theoretical Background

### What is a Multilayer Perceptron?

A multilayer perceptron is a class of feedforward artificial neural network consisting of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.

### Key Components

1. **Neurons and Layers**: 
   - **Input Layer**: Receives the features (e.g., cell nucleus characteristics)
   - **Hidden Layers**: Process the input using weighted connections
   - **Output Layer**: Produces the classification result (malignant or benign)

2. **Forward Propagation**:
   - Each neuron computes a weighted sum of its inputs: `Z = W·X + b`
   - An activation function is applied to introduce non-linearity: `A = σ(Z)`
   - For binary classification, sigmoid activation is often used: `σ(Z) = 1/(1+e^(-Z))`
   - Information flows from the input layer through the hidden layers to the output layer

3. **Loss Function**:
   - For binary classification, we use binary cross-entropy:
   - `E = -1/m * Σ[y·log(p) + (1-y)·log(1-p)]`
   - Where `m` is the number of examples, `y` is the true label, and `p` is the predicted probability

4. **Backpropagation**:
   - Calculates the gradient of the loss function with respect to each weight
   - Propagates the error backward from the output layer to the input layer
   - Updates weights to minimize the loss function

5. **Gradient Descent**:
   - Updates the weights using the calculated gradients: `W = W - learning_rate * dW`
   - The learning rate determines the step size of weight updates

### The Mathematics Behind MLP

For a network with layers indexed by `l` (1 to L):
1. **Initialisation of parameters**:

![image](https://github.com/user-attachments/assets/8250955a-8979-4cff-a4ac-de79d4278a9a)

2. **Forward Pass**:

![image](https://github.com/user-attachments/assets/280c90e2-40b7-481c-b0f1-d5885375d274)

   - `Z^(l) = W^(l)·A^(l-1) + b^(l)`
   - `A^(l) = σ(Z^(l))`

4. **Backward Pass**:

![image](https://github.com/user-attachments/assets/0aa9f320-dc77-4a73-bb3d-5f92a25d9b86)

   - Output layer error: `dZ^(L) = A^(L) - Y`
   - Hidden layer error: `dZ^(l) = (W^(l+1))^T·dZ^(l+1) * σ'(Z^(l))`
   - Weight gradients: `dW^(l) = 1/m * dZ^(l)·(A^(l-1))^T`
   - Bias gradients: `db^(l) = 1/m * Σ(dZ^(l))`

6. **Parameter Update**:
   - `W^(l) = W^(l) - learning_rate * dW^(l)`
   - `b^(l) = b^(l) - learning_rate * db^(l)`

## Project Implementation

This implementation follows the theoretical framework described above, coding a multilayer perceptron from scratch using NumPy.

### Key Features

- Custom implementation of forward and backward propagation
- Configurable number and size of hidden layers
- Binary cross-entropy loss function
- Performance visualization (loss and accuracy plots)
- Dataset preprocessing capabilities

### Code Structure

The project consists of four main components:

1. **Data Preprocessing** (`preprocessing.py`):
   - Reads and formats the Wisconsin Breast Cancer dataset
   - Performs feature scaling and label encoding
   - Splits data into training and validation sets

2. **Model Implementation** (`mlp_classe.py`):
   - Contains the `Multi_layer_perceptron` class with methods for:
     - Network initialization
     - Forward propagation
     - Loss calculation
     - Backpropagation
     - Gradient descent optimization
     - Prediction and evaluation

3. **Training Script** (`train.py`):
   - Loads preprocessed data
   - Initializes and trains the model
   - Saves model weights

4. **Prediction Script** (`predict.py`):
   - Loads a trained model
   - Makes predictions on test data
   - Evaluates model performance

### Implementation Details

The `Multi_layer_perceptron` class handles the entire neural network lifecycle:

- **Initialization**: Sets up the network architecture and initializes weights randomly
- **Forward Propagation**: Computes activations layer by layer
- **Backpropagation**: Calculates gradients for weight updates
- **Optimization**: Updates weights using gradient descent
- **Evaluation**: Computes loss and accuracy metrics

## Usage

### Prerequisites

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- tqdm

### Setup and Installation

```bash
git clone https://github.com/yourusername/multilayer-perceptron.git
cd multilayer-perceptron
pip install -r requirements.txt
```

### Running the Project

1. **Preprocess the data**:
```bash
python preprocessing.py data.csv
```

2. **Train the model**:
```bash
python train.py
```

3. **Make predictions**:
```bash
python predict.py
```

### Customizing the Model 

You can modify the model architecture by changing parameters in `train.py`:

```python
mlp = Multi_layer_perceptron(
    X_train, 
    y_train, 
    X_valid, 
    y_valid, 
    hidden_layers=(24, 24, 24),  # Customize layer sizes
    learning_rate=0.0314,        # Adjust learning rate
    num_iterations=5000          # Set training iterations
)
```

## Results and Visualization

The training process generates:
- Loss curves showing training and validation loss over time
- Accuracy curves showing training and validation accuracy over time
- A classification report with precision, recall, and F1-score

## Future Improvements

Potential enhancements for this implementation:
- Additional activation functions (ReLU, tanh)
- Regularization techniques (L1, L2, dropout)
- Advanced optimizers like Adam or RMSprop
- Learning rate scheduling
- K-fold cross-validation
