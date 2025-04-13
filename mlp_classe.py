from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Multi_layer_perceptron:
    def __init__(self, X_train, y_train, X_test, y_test, hidden_layers=(32, 32, 32), learning_rate=0.01, num_iterations=10000):
        # Initialisation des paramètres
        self.X = X_train.T
        self.y = y_train.reshape(1, -1)
        self.X_test = X_test.T
        self.y_test = y_test.reshape(1, -1)
        self.X = self.X.reshape(self.X.shape[0], -1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], -1)    
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        
        if isinstance(hidden_layers, int):
            dimensions = [hidden_layers]
        else:
            dimensions = list(hidden_layers)
            
        dimensions.insert(0, self.X.shape[0])  
        dimensions.append(self.y.shape[0])    
        self.parametre = self.initialisation(dimensions)
        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []

    def initialisation(self, dimensions):
        parametre = {}
        C = len(dimensions)
        for c in range(1,C):
            parametre['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
            parametre['b' + str(c)] = np.random.randn(dimensions[c], 1)
        return parametre

    def forward_propagation(self, X=None):
        if X is None:
            X = self.X
        parametre = self.parametre
        C = len(self.parametre) // 2
        activations = {'A0': X}
        for c in range(1, C + 1):
            Z = parametre['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametre['b' + str(c)]
            Z = np.clip(Z, -500, 500)
            activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
        return activations

    def log_loss(self, A, y):
        m = y.shape[1]  
        A = np.clip(A, 1e-15, 1 - 1e-15)  # Prevent log(0)
        erreur = (-1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        return erreur

    def back_propagation(self, activations): 
        # Calcul des gradients (dérivées)
        m = self.y.shape[1]
        C = len(self.parametre) // 2 
        parametre = self.parametre
        dZ = activations['A' + str(C)] - self.y
        gradients = {}
        for c in reversed(range(1, C + 1)):
            gradients['dW' + str(c)] = (1/m) * np.dot(dZ, activations['A' + str(c - 1)].T) 
            gradients['db' + str(c)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(parametre['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])  # Fix: use activations

        return gradients

    def update(self, gradient, learning_rate):
        C = len(self.parametre) // 2
        parametre = self.parametre
        for c in range(1, C + 1):
            parametre['W' + str(c)] = parametre['W' + str(c)] - learning_rate * gradient['dW' + str(c)]
            parametre['b' + str(c)] = parametre['b' + str(c)] - learning_rate * gradient['db' + str(c)]

        return parametre
    
    def predict(self, X=None):
        if X is None:
            X = self.X 
        activation = self.forward_propagation(X)
        C = len(self.parametre) // 2 
        return activation['A' + str(C)] >= 0.5

    def cost_plot(self):
        # Tracer l'évolution de la fonction de coût
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_train, label='Train Loss')
        plt.plot(self.loss_test, label='Test Loss')
        plt.xlabel("Nombre itérations")
        plt.ylabel("Cost")
        plt.legend()
        plt.title("Evolution de la fonction de perte")
        plt.show()
    
    def acc_plot(self):
        # Tracer l'évolution de l'accuracy'
        plt.figure(figsize=(10, 6))
        plt.plot(self.acc_train, label='Train Accuracy')
        plt.plot(self.acc_test, label='Test Accuracy')
        plt.xlabel("Nombre itérations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Evolution de l'accuracy")
        plt.show()

    def gradient_descent(self):
        # Descente de gradient pour l'entraînement du modèle
        for i in tqdm(range(self.num_iterations)):
            if i % 100 == 0:
                C = len(self.parametre) // 2
                # Train
                activations_train = self.forward_propagation(self.X)
                self.loss_train.append(self.log_loss(activations_train['A' + str(C)], self.y))
                y_pred_train = self.predict(self.X)
                self.acc_train.append(accuracy_score(self.y.flatten(), y_pred_train.flatten()))
                
                # Test
                activations_test = self.forward_propagation(self.X_test)
                self.loss_test.append(self.log_loss(activations_test['A' + str(C)], self.y_test))
                y_pred_test = self.predict(self.X_test)
                self.acc_test.append(accuracy_score(self.y_test.flatten(), y_pred_test.flatten()))
            
            # Entrainement
            activations = self.forward_propagation()
            gradients = self.back_propagation(activations) 
            self.update(gradients, self.learning_rate)
        
        # Afficher resultat
        self.cost_plot()
        self.acc_plot()
        y_pred = self.predict(self.X_test)
        print(classification_report(self.y_test.flatten(), y_pred.flatten())) 