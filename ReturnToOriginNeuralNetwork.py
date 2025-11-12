import numpy as np

from HeadDirectionNeuralNetwork import HeadDirectionNeuralNetwork
from ReturnToOriginCell import ReturnToOriginCell


class ReturnToOriginNeuralNetwork:
    """Return to origin neural network."""

    def __init__(self, neuron_count, threshold):
        self.neuron_count = neuron_count
        self.threshold = threshold
        self.cells_out_collection = [ReturnToOriginCell(i, threshold) for i in range(neuron_count)]  # Output layer
        self.cells_hidden_collection = [ReturnToOriginCell(i, threshold) for i in
                                        range(neuron_count + int(threshold) - 1)]  # Hidden layer
        self.cells_input_collection = [ReturnToOriginCell(i, threshold) for i in range(neuron_count)]  # Input layer
        self.synaptic_weight = 1.0

        self.connect_with_fixed_points(self.synaptic_weight)

    def connect_with_fixed_points(self, initial_weight):
        """Establish connections between neurons with fixed weights."""
        for output_cell in self.cells_out_collection:
            for i in range(output_cell.id, len(self.cells_hidden_collection)):
                output_cell.synapses_all[self.cells_hidden_collection[i]] = initial_weight

    def simulate(self, inputs):
        """Simulate the neural network with given inputs."""
        output_array = [0] * len(self.cells_hidden_collection)
        placeholder_length = len(self.cells_hidden_collection) - self.neuron_count

        # Map inputs to hidden layer, handling circular input
        for i in range(len(inputs)):
            output_array[i + placeholder_length] = inputs[i]
        for i in range(placeholder_length):
            output_array[i] = inputs[self.neuron_count - placeholder_length + i]

        # Apply max activation inhibition
        output_array = self.max_activate_inhibit(output_array)

        # Predict output for each output neuron
        output = [self.cells_out_collection[i].predict(output_array) for i in range(self.neuron_count)]
        return output

    def max_activate_inhibit(self, arr):
        """Inhibit non-maximum continuous activation segments."""
        if not arr or len(arr) == 0:
            return []

        max_count = 0  # Length of longest continuous 1s
        max_start = -1  # Start index of longest continuous 1s
        current_count = 0  # Current continuous 1s count
        current_start = -1  # Current start index

        # Find the longest continuous segment of 1s
        for i in range(len(arr)):
            if arr[i] == 1:
                if current_count == 0:
                    current_start = i
                current_count += 1
                if current_count > max_count:
                    max_count = current_count
                    max_start = current_start
            else:
                current_count = 0
                current_start = -1

        # Set all 1s outside the longest segment to 0
        for i in range(len(arr)):
            if arr[i] == 1 and (i < max_start or i >= max_start + max_count):
                arr[i] = 0
        return arr


if __name__ == "__main__":
    ring = HeadDirectionNeuralNetwork(10, 100, 1.0, 1.0, 1, 2)
    encode = ring.simulate([9, 0])  # Actual value is one more than the last position
    encode_vector = [0] * len(encode[encode.shape[0] - 1])
    threshold_value = 0
    for i in range(len(encode[encode.shape[0] - 1])):
        if encode[encode.shape[0] - 1][i]:  # If value is True, set corresponding vector to 1
            encode_vector[i] = 1
            threshold_value += 1

    ellipsoid = ReturnToOriginNeuralNetwork(10, threshold_value)
    result_state = ellipsoid.simulate(encode_vector)
    print(result_state)
    # Excitatory neuron vector, to be converted into new neural positions