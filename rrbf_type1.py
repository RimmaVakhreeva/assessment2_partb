import random
import math


# Define the RRBF_type1 Network class
class RRBF_type1_Network:
    def __init__(self, num_neurons, num_inputs):
        self.centers = [[random.gauss(0, 1) for _ in range(num_inputs)] for _ in range(num_neurons)]
        self.stds = [random.random() for _ in range(num_neurons)]
        self.weights = [random.gauss(0, 1) for _ in range(num_neurons)]

    # Define the Gaussian function
    def gaussian_function(self, x, m, s):
        return math.exp(-sum((x - m) ** 2 for x, m in zip(x, m)) / (2 * s ** 2))

    def compute_phi(self, x):
        return [self.gaussian_function(x, self.centers[i], self.stds[i]) for i in range(len(self.centers))]

    def forward_pass(self, x):
        phi = self.compute_phi(x)
        output = sum(w * p for w, p in zip(self.weights, phi))
        return output

    def compute_gradients(self, x, y, eta):
        gradients_delta = [0] * len(self.stds)
        gradients_m = [[0] * len(m) for m in self.centers]
        gradient_w = [0] * len(self.weights)

        phi = self.compute_phi(x)
        output = self.forward_pass(x)
        error = output - y

        for i in range(len(self.stds)):
            phi_i = phi[i]
            diff = [x - m for x, m in zip(x, self.centers[i])]
            gradients_delta[i] = eta * error * self.weights[i] * phi_i * sum(d ** 2 for d in diff) / self.stds[i] ** 3
            gradients_m[i] = [eta * error * self.weights[i] * phi_i * d / self.stds[i] ** 2 for d in diff]
            # Weight update
            gradient_w[i] = eta * error * phi_i

        return gradients_delta, gradients_m, gradient_w

    def train(self, x_train, y_train, eta, epochs):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                gradients_delta, gradients_m, gradient_w = self.compute_gradients(x, y, eta)

                self.stds = [s - g for s, g in zip(self.stds, gradients_delta)]
                self.centers = [[c - g for c, g in zip(center, grad_m)] for center, grad_m in
                                zip(self.centers, gradients_m)]
                self.weights = [w - g for w, g in zip(self.weights, gradient_w)]


# Example usage
num_neurons = 10
num_inputs = 5
rrbf_network = RRBF_type1_Network(num_neurons, num_inputs)

# Placeholder training data
x_train = [[random.gauss(0, 1) for _ in range(num_inputs)] for _ in range(100)]  # 100 samples
y_train = [random.gauss(0, 1) for _ in range(100)]  # 100 target values
eta = 0.01  # Learning rate
epochs = 10  # Number of epochs for training

# Train the network
rrbf_network.train(x_train, y_train, eta, epochs)

# After training, the network's parameters have been updated
print("Updated standard deviations:", rrbf_network.stds)
print("Updated centers:", rrbf_network.centers)
print("Updated weights:", rrbf_network.weights)
