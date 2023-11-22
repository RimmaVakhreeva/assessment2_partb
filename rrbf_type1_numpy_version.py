import numpy as np


class RRBF_type1_Network:
    def __init__(self, num_neurons, num_inputs):
        self.centers = np.random.randn(num_neurons, num_inputs)
        self.stds = np.random.rand(num_neurons, 1)  # stds as column vector for broadcasting
        self.weights = np.random.randn(num_neurons)

    def gaussian_function(self, x, m, s):
        return np.exp(-((x - m) ** 2) / (2 * s ** 2))

    def compute_phi(self, x):
        phi = np.array([self.gaussian_function(x, m, s) for m, s in zip(self.centers, self.stds)])
        return phi

    def forward_pass(self, x):
        phi = self.compute_phi(x)
        output = np.dot(self.weights, phi)
        return output

    def compute_gradients(self, x, y, eta):
        gradients_delta = np.zeros_like(self.stds)
        gradients_m = np.zeros_like(self.centers)
        phi = self.compute_phi(x)
        error = self.forward_pass(x) - y

        for i in range(len(self.stds)):
            phi_i = self.gaussian_function(x, self.centers[i], self.stds[i])
            gradients_delta[i] = eta * error * self.weights[i] * phi_i * (x - self.centers[i]) ** 2 / self.stds[i] ** 3
            gradients_m[i] = eta * error * self.weights[i] * phi_i * (x - self.centers[i]) / self.stds[i] ** 2

        # Weight update
        gradient_w = eta * error * phi
        return gradients_delta, gradients_m, gradient_w

    def train(self, x_train, y_train, eta, epochs):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                gradients_delta, gradients_m, gradient_w = self.compute_gradients(x, y, eta)
                self.stds -= gradients_delta
                self.centers -= gradients_m
                self.weights -= gradient_w


# Example usage
num_neurons = 10
num_inputs = 5
rrbf_network = RRBF_type1_Network(num_neurons, num_inputs)

# Placeholder training data
x_train = np.random.randn(100, num_inputs)  # 100 samples, each with 'num_inputs' features
y_train = np.random.randn(100)  # 100 target values
eta = 0.01  # Learning rate
epochs = 10  # Number of epochs for training

# Train the network
rrbf_network.train(x_train, y_train, eta, epochs)

# After training, the network's centers, std deviations, and weights have been updated
print("Updated standard deviations:", rrbf_network.stds.flatten())
print("Updated centers:", rrbf_network.centers)
print("Updated weights:", rrbf_network.weights)
