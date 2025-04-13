from mlp_classe import Multi_layer_perceptron
import numpy as np
from sklearn.metrics import classification_report

X_valid = np.load("X_valid.npy")
y_valid = np.load("y_valid.npy")


model = Multi_layer_perceptron(X_valid, y_valid, X_valid, y_valid)
model.parametre = np.load("model_weights.npy", allow_pickle=True).item()

y_pred = model.predict(X_valid.T)
print(classification_report(y_valid, y_pred.flatten()))
