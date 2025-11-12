class HeadDirectionCell:
    """Creates a head direction neuron."""

    def __init__(self, id_, angle, time, activated_threshold):
        self.id = id_  # Neuron's ID
        self.angle = angle
        self.time = time
        self.exc_synapses_ps = {}  # Map for excitatory synapses from previous iteration
        self.inh_synapses_ps = {}  # Map for inhibitory synapses from previous iteration
        self.exc_synapses_all = {}  # Map for all excitatory connections for propagation
        self.inh_synapses_all = {}  # Map for all inhibitory connections for propagation
        self.activated = False
        self.activated_threshold = activated_threshold

    def step(self):
        """Process excitatory and inhibitory inputs to determine activation."""
        total_input = 0.0

        # Process excitatory synapses
        for pre_synaptic_cell, _ in self.exc_synapses_ps.items():
            weight = pre_synaptic_cell.exc_synapses_all.get(self, 0.0)
            total_input += weight * 1  # Neuron input is 1, value is weight

        # Process inhibitory synapses
        for pre_synaptic_cell, _ in self.inh_synapses_ps.items():
            weight = pre_synaptic_cell.inh_synapses_all.get(self, 0.0)
            total_input -= weight * 1  # Neuron input is 1, value is weight

        if total_input >= self.activated_threshold:
            self.activated = True
            self.propagate_excite()
        else:
            self.activated = False

        # Clear synapses for the next iteration
        self.exc_synapses_ps.clear()
        self.inh_synapses_ps.clear()

    def propagate_excite(self):
        """Propagate excitation to connected neurons."""
        for cell, weight in self.exc_synapses_all.items():
            cell.exc_synapses_ps[self] = weight

        for cell, weight in self.inh_synapses_all.items():
            cell.inh_synapses_ps[self] = weight

    # Getter and setter methods converted to Python properties where necessary
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def exc_synapses(self):
        return self.exc_synapses_ps

    @exc_synapses.setter
    def exc_synapses(self, value):
        self.exc_synapses_ps = value

    @property
    def inh_synapses(self):
        return self.inh_synapses_ps

    @inh_synapses.setter
    def inh_synapses(self, value):
        self.inh_synapses_ps = value

    @property
    def activated(self):
        return self._activated

    @activated.setter
    def activated(self, value):
        self._activated = value
        if value:
            self.propagate_excite()

    @property
    def exc_synapses_all(self):
        return self._exc_synapses_all

    @exc_synapses_all.setter
    def exc_synapses_all(self, value):
        self._exc_synapses_all = value

    @property
    def inh_synapses_all(self):
        return self._inh_synapses_all

    @inh_synapses_all.setter
    def inh_synapses_all(self, value):
        self._inh_synapses_all = value
