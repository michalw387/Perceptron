from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, epochs=100, learning_rate=0.001) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate

    def reshape_data(self):
        self.images = self.images.reshape(len(self.images), -1)

    def standardizate_data(self):
        divisor = 255
        self.images = [x / divisor for x in self.images]

    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.images, self.labels
        )

    def initialize_variables(self):
        self.weights = np.zeros(len(self.images[0]), dtype="float64")
        self.bias = 0.0

    @staticmethod
    def activation_fun(z):
        # sigmoid function
        return 1.0 / (1.0 + np.exp(-z))

    def cost_fun(self, A, Y, m):
        # binary cross-entropy
        epsilon = 1e-9  # Small constant to prevent log(0)
        return (
            -np.sum(Y * np.log(A + epsilon) + (1.0 - Y) * np.log(1.0 - A + epsilon)) / m
        )

    def feedforward(self, a):
        return self.activation_fun((self.weights @ a) + self.bias)

    def learn(self):
        self.m = len(self.x_train)
        self.costs = []
        for i in range(self.epochs):

            A = self.feedforward(np.array(self.x_train).T)

            self.costs.append(self.cost_fun(A, self.y_train, self.m))

            if i % 10 == 0:
                print(f"Cost in {i} iteration: {self.costs[-1]}")

            # calculate gradients
            dw = (np.array(self.x_train).T @ (A - self.y_train)) / self.m
            db = np.sum(A - self.y_train) / self.m

            # update weights and bias
            self.weights = self.weights - dw * self.learning_rate
            self.bias = self.bias - db * self.learning_rate

    def plot_cost(self):
        plt.figure(figsize=(10, 6))
        plt.title("Cost function")
        plt.plot(
            range(len(self.costs)),
            self.costs,
            color="orange",
            label="cost",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Cost function")
        plt.grid()
        plt.show()

    def predict(self, X):
        X = np.array(X)
        m = len(X)
        A = self.activation_fun((self.weights @ X.T) + self.bias)

        Y_pred = [0] * m
        for i, a in enumerate(A):
            if a < 0.5 and a > 0:
                Y_pred[i] = 0
            else:
                Y_pred[i] = 1
        return Y_pred

    def test(self):
        Y_pred_train = self.predict(self.x_train)
        Y_pred_test = self.predict(self.x_test)

        print(
            "Train accuracy: {} %".format(
                round(100 - np.mean(np.abs(Y_pred_train - self.y_train)) * 100, 2)
            )
        )
        print(
            "Test accuracy: {} %".format(
                round(100 - np.mean(np.abs(Y_pred_test - self.y_test)) * 100, 2)
            )
        )

        f1_test = f1_score(self.y_test, Y_pred_test)
        print(f"Test F1 score = {round(f1_test, 2)}")

    def start_perceptron(self):
        self.images = np.load("data/images.npy")
        self.labels = np.load("data/labels.npy")
        self.reshape_data()
        self.standardizate_data()
        self.split_data()
        self.initialize_variables()

        self.learn()
        self.plot_cost()


p = Perceptron(epochs=200, learning_rate=0.001)

p.start_perceptron()
p.test()
