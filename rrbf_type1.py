import math
import random


class RRBFType1:
    """
    Represents a Radial Basis Function (RBF) Network of type 1.

    This class creates a Radial Basis Function Network with a specified number of neurons
    and input dimensions. It initializes centers, standard deviations, and weights
    for each neuron randomly.

    Attributes:
        centers (list): Centers of the RBF neurons, initialized randomly.
        stds (list): Standard deviations of the RBF neurons, initialized randomly.
        weights (list): Weights of the RBF neurons, initialized randomly.
        P (list): Matrix used in recursive least squares algorithm, initialized as None.
        alpha (float): Regularization parameter for recursive least squares.
        backward_cache (dict): Cache to store intermediate values for the backward pass.
    """


    def __init__(self, num_neurons, max_num_inputs):
        """
        Initializes the RRBFType1 instance.

        Args:
            num_neurons (int): Number of neurons in the RBF network.
            max_num_inputs (int): Maximum number of inputs that the network can handle.

        The centers and weights are initialized using a Gaussian distribution,
        while the standard deviations are initialized using a uniform distribution.
        """
        # Initialize centers with Gaussian distribution
        self.centers = [[random.gauss(0, 1) for _ in range(max_num_inputs)] for _ in range(num_neurons)]

        # Initialize standard deviations with uniform distribution
        self.stds = [random.random() for _ in range(num_neurons)]

        # Initialize weights with Gaussian distribution
        self.weights = [random.gauss(0, 1) for _ in range(num_neurons)]

        # Initialize P matrix for recursive least squares as None
        self.P = None

        # Set alpha, a regularization parameter for recursive least squares
        self.alpha = 1000

        # Initialize cache for backward pass
        self.backward_cache = dict(
            x=None,
            phi=None,
            y=None
        )

    def _gaussian_function(self, x, c, s):
        """
        Computes the Gaussian function for a given input, center, and standard deviation.

        Args:
            x (list of float): Input vector.
            c (list of float): Center vector of the RBF neuron.
            s (float): Standard deviation of the RBF neuron.

        Returns:
            float: Output of the Gaussian function.

        The Gaussian function is computed as exp(-||x - c||^2 / (2 * s^2)).
        """
        # Compute Gaussian function value
        return math.exp(-sum((x - c) ** 2 for x, c in zip(x, c)) / (2 * s ** 2))

    def _compute_phi(self, x):
        """
        Computes the output of the Gaussian functions for all neurons.

        Args:
            x (list of float): The input vector.

        Returns:
            list of float: The outputs of the Gaussian functions for each neuron.
        """
        # Compute output of Gaussian function for each neuron
        return [self._gaussian_function(x, self.centers[i], self.stds[i]) for i in range(len(self.centers))]

    def _recursive_least_squares(self, phi, y):
        """
        Performs the Recursive Least Squares algorithm for weight update.

        Args:
            phi (list of float): Outputs of Gaussian functions for current input.
            y (float): Target output value for the current input.

        Returns:
            list of float: Updated weights after applying the RLS algorithm.

        This method updates the weights of the network using the RLS algorithm.
        It initializes the P matrix if it's not already initialized.
        """

        # Initialize P matrix if it is the first call
        if self.P is None:
            self.P = [[self.alpha if i == j else 0 for j in range(len(self.weights))]
                      for i in range(len(self.weights))]

        # Transpose phi vector for matrix operations
        phi_vector_transpose = [[x] for x in phi]

        # Compute P_phi for RLS update
        P_phi = [sum(P_i * phi_j for P_i, phi_j in zip(P_row, phi)) for P_row in self.P]

        # Calculate the scalar used in updating K and weights
        temp = 1 / (sum(phi_t[0] * P_phi_i for phi_t, P_phi_i in zip(phi_vector_transpose, P_phi)) + self.alpha)

        # Compute k vector
        k = [temp * P_phi_i for P_phi_i in P_phi]

        # Update the weights using the RLS algorithm
        self.weights = [w - k_i * (sum(w_p * phi_p for w_p, phi_p in zip(self.weights, phi)) - y)
                        for w, k_i in zip(self.weights, k)]

        # Update the P matrix
        for i in range(len(self.P)):
            for j in range(len(self.P[i])):
                self.P[i][j] = self.P[i][j] - k[i] * phi_vector_transpose[j][0]

        # Return the updated weights
        return self.weights

    def forward(self, x):
        """
        Performs the forward pass of the RBF network.

        Args:
            x (list of float): The input vector.

        Returns:
            float: The output of the network for the given input.

        This method computes the output of the network by applying the RBF
        Gaussian functions and then summing up the weighted outputs.
        """

        # Compute phi values using Gaussian functions
        phi = self._compute_phi(x)

        # Compute the network output as a weighted sum of phi values
        y = sum(w * p for w, p in zip(self.weights, phi))

        # Store input, phi, and output in the cache for the backward pass
        self.backward_cache['x'] = x
        self.backward_cache['phi'] = phi
        self.backward_cache['y'] = y

        # Return the computed output
        return y

    def backward(self, grad_output):
        """
        Performs the backward pass of the RBF network.

        Args:
            grad_output (float): The gradient of the loss function with respect to the output.

        Returns:
            tuple: Gradients with respect to standard deviations, centers, weights, and input x.

        This method computes the gradients of the loss function with respect to
        the network's parameters and input by applying the chain rule.
        Raises an error if called before the forward pass.
        """

        # Check if forward pass data is available in the cache
        if self.backward_cache['phi'] is None:
            raise ValueError("Forward pass must be called before backward pass.")

        # Initialize gradients for standard deviations, centers, and weights
        gradients_delta = [0] * len(self.stds)
        gradients_center = [[0] * len(m) for m in self.centers]
        gradient_weights = [0] * len(self.weights)

        # Retrieve cached input, phi, and output values
        x = self.backward_cache['x']
        phi = self.backward_cache['phi']
        y = self.backward_cache['y']

        # Compute gradients for each neuron
        for i, (std, center, phi_i) in enumerate(zip(self.stds, self.centers, phi)):
            # Difference between input and center
            diff = [xi - ci for xi, ci in zip(x, center)]

            # Gradient with respect to standard deviation
            gradients_delta[i] = self.weights[i] * phi_i * sum(d ** 2 for d in diff) / (std ** 3)
            # Gradient with respect to center for each dimension
            gradients_center[i] = [self.weights[i] * phi_i * d / (std ** 2) for d in diff]
            # Update weights using the recursive least squares method
            gradient_weights[i] = self._recursive_least_squares(phi, y)

        # Initialize gradient with respect to input x
        gradients_x = [0] * len(x)
        # Compute gradient for each input dimension
        for i, (center, std) in enumerate(zip(self.centers, self.stds)):
            weight = self.weights[i]
            phi_i = phi[i]
            for j, x_j in enumerate(x):
                # Difference between input and center for the j-th dimension
                diff = x_j - center[j]
                # Gradient of the Gaussian function with respect to x_j
                grad_gaussian_x_j = -diff / (std ** 2) * phi_i
                # Accumulate gradients for each input dimension
                gradients_x[j] += weight * grad_gaussian_x_j

        # Return gradients with respect to standard deviations, centers, weights, and input x
        return gradients_delta, gradients_center, gradient_weights, gradients_x
