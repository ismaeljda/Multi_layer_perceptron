import numpy as np
from mlp_classe import Multi_layer_perceptron

X_train = np.load("X_train.npy")
X_valid = np.load("X_valid.npy")
y_train = np.load("y_train.npy")
y_valid = np.load("y_valid.npy")

mlp = Multi_layer_perceptron(X_train, y_train, X_valid, y_valid, hidden_layers=(24, 24, 24), learning_rate=0.0314, num_iterations=5000)
mlp.gradient_descent()
np.save("model_weights.npy", mlp.parametre)