class ReturnToOriginCell:
    """Neuron that converts activation values to origin values."""

    def __init__(self, id_, activated_threshold):
        self.id = id_
        self.synapses_all = {}  # Map for all excitatory connections for propagation
        self.activated = False
        self.activated_threshold = activated_threshold

    def forward(self, input_array):
        """Forward propagation, processes all inputs."""
        total_input = 0.0
        for pre_synaptic_cell, weight in self.synapses_all.items():
            index = pre_synaptic_cell.id
            input_value = input_array[index]
            total_input += input_value * weight  # Neuron input multiplied by weight
        return total_input

    def predict(self, input_array):
        """Predict activation based on threshold."""
        return 1 if self.forward(input_array) >= self.activated_threshold else 0

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