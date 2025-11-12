
import numpy as np

from HeadDirectionCell import HeadDirectionCell


class HeadDirectionNeuralNetwork:
    """Head direction neural network."""

    def __init__(self, neuron_counts, epoch_time, initial_inh_weight, initial_exc_weight, exc_neuron_count,
                 inh_neuron_count):
        self.neuron_counts = neuron_counts
        self.initial_inh_weight = initial_inh_weight
        self.initial_exc_weight = initial_exc_weight
        self.exc_neuron_count = exc_neuron_count
        self.inh_neuron_count = inh_neuron_count
        self.epoch_time = epoch_time

        self.neuron_cell_collection = []
        activated_threshold = (exc_neuron_count - 1) * initial_exc_weight  # Activation threshold
        if activated_threshold == 0:
            activated_threshold += 0.00000000001

        for i in range(neuron_counts):
            self.neuron_cell_collection.append(
                HeadDirectionCell(i, 360.0 / neuron_counts * i, i, activated_threshold)
            )

        self.connect_with_fixed_points(initial_inh_weight, initial_exc_weight)

    def connect_with_fixed_points(self, initial_inh_weight, initial_exc_weight):
        """Establish excitatory and inhibitory connections between neurons."""
        for neuron_cell in self.neuron_cell_collection:
            # Establish excitatory connections
            for i in range(1, self.exc_neuron_count + 1):
                left_neighbor_neuron_index = neuron_cell.id - i
                if left_neighbor_neuron_index < 0:
                    left_neighbor_neuron_index += self.neuron_counts
                neuron_cell.exc_synapses_all[
                    self.neuron_cell_collection[left_neighbor_neuron_index]] = initial_exc_weight

                right_neighbor_neuron_index = neuron_cell.id + i
                if right_neighbor_neuron_index >= self.neuron_counts:
                    right_neighbor_neuron_index -= self.neuron_counts
                neuron_cell.exc_synapses_all[
                    self.neuron_cell_collection[right_neighbor_neuron_index]] = initial_exc_weight

            # Establish inhibitory connections
            for i in range(self.exc_neuron_count + 1, self.exc_neuron_count + self.inh_neuron_count + 1):
                left_neighbor_neuron_index_inh = neuron_cell.id - i
                if left_neighbor_neuron_index_inh < 0:
                    left_neighbor_neuron_index_inh += self.neuron_counts
                neuron_cell.inh_synapses_all[
                    self.neuron_cell_collection[left_neighbor_neuron_index_inh]] = initial_inh_weight

                right_neighbor_neuron_index_inh = neuron_cell.id + i
                if right_neighbor_neuron_index_inh >= self.neuron_counts:
                    right_neighbor_neuron_index_inh -= self.neuron_counts
                neuron_cell.inh_synapses_all[
                    self.neuron_cell_collection[right_neighbor_neuron_index_inh]] = initial_inh_weight

    def simulate(self, provoke_ids):
        """Simulate the neural network for the specified number of epochs."""
        neuron_cell_collection_state = np.full((self.epoch_time, self.neuron_counts), False, dtype=bool)

        for t in range(1, self.epoch_time + 1):
            """
            # Commented out weight update code
            for neuron_cell in self.neuron_cell_collection:
                if t != 1 and neuron_cell_collection_state[t-2][neuron_cell.id] is not None and neuron_cell_collection_state[t-2][neuron_cell.id]:
                    for neuron_cell_pre in self.neuron_cell_collection:
                        exc_synapses_from_pre_cell = neuron_cell_pre.exc_synapses_all
                        for head_direction_cell_post in exc_synapses_from_pre_cell.keys():
                            if head_direction_cell_post.id == neuron_cell.id:
                                if neuron_cell_collection_state[t-2][neuron_cell_pre.id] is not None and neuron_cell_collection_state[t-2][neuron_cell_pre.id]:
                                    new_weight = exc_synapses_from_pre_cell[neuron_cell] + 0.04
                                    if new_weight > 1:
                                        new_weight = 1.0
                                    neuron_cell_pre.exc_synapses_all[neuron_cell] = new_weight

            for neuron_cell in self.neuron_cell_collection:
                if t != 1 and neuron_cell_collection_state[t-2][neuron_cell.id] is not None and not neuron_cell_collection_state[t-2][neuron_cell.id]:
                    for neuron_cell_pre in self.neuron_cell_collection:
                        inh_synapses_from_pre_cell = neuron_cell_pre.inh_synapses_all
                        for head_direction_cell_post in inh_synapses_from_pre_cell.keys():
                            if head_direction_cell_post.id == neuron_cell.id:
                                if neuron_cell_collection_state[t-2][neuron_cell_pre.id] is not None and neuron_cell_collection_state[t-2][neuron_cell_pre.id]:
                                    new_weight = inh_synapses_from_pre_cell[neuron_cell] + 0.04
                                    if new_weight > 1:
                                        new_weight = 1.0
                                    neuron_cell_pre.inh_synapses_all[neuron_cell] = new_weight
            """
            # Perform simulation step
            for neuron_cell in self.neuron_cell_collection:
                if t == 1 and neuron_cell.id == 0:
                    for i in provoke_ids:
                        self.neuron_cell_collection[i].activated = True
                neuron_cell.step()
                neuron_cell_collection_state[t - 1][neuron_cell.id] = neuron_cell.activated

        return neuron_cell_collection_state


if __name__ == "__main__":
    ring = HeadDirectionNeuralNetwork(360, 1000, 1.0, 1.0, 4, 4)
    result_state = ring.simulate([29, 30, 31, 32, 33, 34, 35])
    # Excitatory neuron vector, to be converted into new neural positions