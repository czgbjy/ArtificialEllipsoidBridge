class BridgeCell:
    """Neuron for bridging inputs from two sources."""

    def __init__(self, id_):
        self.id = id_
        self.synapses_all = {}  # Dictionary for all excitatory connections

    def forward(self, input_left, input_right):
        """Forward propagation, processes inputs from left and right sources."""
        total_input = 0.0
        all_input_synapses = list(self.synapses_all.items())  # Convert dictionary to list of (cell, weight) tuples

        if len(all_input_synapses) == 2:  # Ensure exactly 2 inputs for correct structure
            # First input
            index_one = all_input_synapses[0][0].id
            input_value_one = input_left[index_one]
            weight_one = all_input_synapses[0][1]
            total_input += input_value_one * weight_one

            # Second input
            index_two = all_input_synapses[1][0].id
            input_value_two = input_right[index_two]
            weight_two = all_input_synapses[1][1]
            total_input += input_value_two * weight_two

        return total_input

    def predict(self, input_left, input_right):
        """Predict output based on inputs from left and right sources."""
        return self.forward(input_left, input_right)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def synapses_all(self):
        return self._synapses_all

    @synapses_all.setter
    def synapses_all(self, value):
        self._synapses_all = value
