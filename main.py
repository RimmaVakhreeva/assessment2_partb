import random

from normal_distribution import generate_random_data, kolmogorov_statistic, visualize_data
from rrbf_type1 import RRBFType1


def task_1():
    # Define the number of neurons and input dimensions for the RBF network
    num_neurons = 10
    num_inputs = 5

    # Initialize an instance of RRBFType1 with the specified number of neurons and inputs
    rrbf_type1 = RRBFType1(num_neurons, num_inputs)

    # Generate a random input vector with the specified number of inputs
    # Each input value is randomly drawn from a uniform distribution between 0 and 1
    x = [random.random() for _ in range(num_inputs)]

    # Perform a forward pass of the RBF network with the generated input vector
    # This calculates the network's output for the given input
    forward_value = rrbf_type1.forward(x)

    # Perform a backward pass to compute gradients
    # This calculates the gradients of the loss function
    # with respect to the standard deviations, centers, weights, and input x
    gradients_delta, gradients_center, gradient_weights, gradients_x = rrbf_type1.backward(x)

    # Print the results for inspection

    # Display the forward pass output value
    print(f'Forward value: {forward_value}')

    # Display the calculated gradients with respect to the standard deviations of the RBF neurons
    print(f'Gradients with respect to standard deviations: {gradients_delta}')

    # Display the calculated gradients with respect to the centers of the RBF neurons
    print(f'Gradients with respect to centers: {gradients_center}')

    # Display the calculated gradients with respect to the weights of the RBF neurons
    print(f'Gradients with respect to weights: {gradient_weights}')

    # Display the calculated gradients with respect to the input vector x
    print(f'Gradients with respect to input x: {gradients_x}')


def task_2():
    # Generating 1000 random data points
    data = generate_random_data(1000)

    # Calculating Kolmogorov statistic
    d_max = kolmogorov_statistic(data)
    #print(f'Kolmogorov statistic: {round(d_max, 3)}')

    # Visualizing the data
    visualize_data(data)


if __name__ == "__main__":
    print('Results of the first task:\n')
    task_1()
    print()
    print('Results of the second task:\n')
    task_2()
