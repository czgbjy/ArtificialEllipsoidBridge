import numpy as np
from typing import List
import time

from ComplementNetwork import ComplementNetwork
from HeadDirectionNeuralNetwork import HeadDirectionNeuralNetwork
from ReturnToOriginNeuralNetwork import ReturnToOriginNeuralNetwork
from AccumulatorNeuralNetwork import AccumulatorNeuralNetwork
from BridgeCell import BridgeCell


class BridgeNeuralNetwork:
    def __init__(self, neuron_count: int):
        """
        Initialize the BridgeNeuralNetwork with specified neuron count.
        neuron_count: Number of neurons in the hidden layer; input layers have twice this number.
        """
        self.neuron_count = neuron_count
        self.cells_hidden_collection = [BridgeCell(i) for i in range(neuron_count)]
        self.cells_input_left_bridge_collection = [BridgeCell(i) for i in range(neuron_count)]
        self.cells_input_right_bridge_collection = [BridgeCell(i) for i in range(neuron_count)]
        self.synaptic_weight = 1.0

        self.connect_with_fixed_points(self.synaptic_weight)

    def connect_with_fixed_points(self, initial_weight: float) -> None:
        """
        Connect hidden layer neurons to corresponding left and right bridge input neurons.
        """
        for i in range(self.neuron_count):
            self.cells_hidden_collection[i].synapses_all[self.cells_input_left_bridge_collection[i]] = initial_weight
            self.cells_hidden_collection[i].synapses_all[self.cells_input_right_bridge_collection[i]] = initial_weight

    def simulate(self, inputs: List[int]) -> List[int]:
        """
        Simulate the neural network with given inputs.
        Returns the indices of activated neurons after global inhibition.
        """
        output = [0] * self.neuron_count
        if len(inputs) == 2 * self.neuron_count:
            input_left = inputs[:self.neuron_count]
            input_right = inputs[self.neuron_count:2 * self.neuron_count]
            sum_up = [self.cells_hidden_collection[i].predict(input_left, input_right) for i in
                      range(self.neuron_count)]
            output = self.inhibit_columns_global(self.neuron_count, sum_up)
        else:
            print("输入向量的维数不是输出神经元维数的2倍，结构不支持")
        return output

    def inhibit_columns_global(self, neuron_count: int, overlaps: List[float]) -> List[int]:
        """
        Apply global inhibition to select winning neurons based on overlap scores.
        """
        # Create list of (index, overlap) tuples and sort by overlap (descending) and index (ascending)
        sorted_winner_indices = [
            pair[0] for pair in sorted(
                [(i, overlaps[i]) for i in range(len(overlaps))],
                key=lambda x: x[1]  # 这里相当于Java中的inhibitionComparator
            )
        ]


        # Enforce stimulus threshold
        stimulus_threshold = overlaps[sorted_winner_indices[len(overlaps)-1]]  # Highest overlap value
        start = 0
        while start < len(sorted_winner_indices):
            i = sorted_winner_indices[start]
            if overlaps[i] >= stimulus_threshold:
                break
            start += 1

        return sorted_winner_indices[start:]

def main_new():
    x_dis=2500
    # 角度转换
    estimate_head_direction = (x_dis - 500) / (100 / 9)  # 这是头部的方向角大小
    # 计算数组的最后一个数字
    estimate_head_direction = int(estimate_head_direction)
    last_number = estimate_head_direction - 1
    # 生成数组：从last_number开始，依次减1，共5个数字
    result = []
    current = last_number
    for _ in range(5):
        result.append(current)
        current -= 1
    # 处理小于0的数字，加上360
    result = [x if x >= 0 else x + 360 for x in result]
    result_new=sorted(result)

    ring = HeadDirectionNeuralNetwork(360, 100, 1.0, 1.0, 3, 4)
    encode = ring.simulate(result_new)
    encode_vector = [0] * len(encode[-1])
    threshold_value = 0
    for i, val in enumerate(encode[-1]):
        if val:  # If True, set corresponding vector element to 1
            encode_vector[i] = 1
            threshold_value += 1

    # Initialize and simulate ReturnToOriginNeuralNetwork
    ellipsoid = ReturnToOriginNeuralNetwork(360, threshold_value)
    result_state = ellipsoid.simulate(encode_vector)

    # Reference output for 45 degrees
    result_state_reference_to_out = [1] * 180 + [0] * 180
    complementNetwork = ComplementNetwork(360)
    # 10 变为 190（补码操作）
    converseInput2 = complementNetwork.forward(result_state_reference_to_out)
    # Initialize and compute with AccumulatorNeuralNetwork
    neural_network = AccumulatorNeuralNetwork(len(result_state))
    neural_network.compute(result_state, converseInput2)

    print(f"{list(neural_network.outputs)}\n")
    merged_array = neural_network.outputs

    # Convert double array to int array
    int_result_array = [int(x) for x in merged_array]

    # Initialize and simulate BridgeNeuralNetwork
    bridge_neural_network = BridgeNeuralNetwork(360)
    final_result_index = bridge_neural_network.simulate(int_result_array)

    # Convert indices to final encode vector
    final_encode = [0] * 360
    result_degree = 0
    for i in final_result_index:
        final_encode[final_result_index[i]] = 1
        result_degree += 1

    print(result_degree)
def main_two():
    # 原始 Java 代码中的数组
    encodeNum1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 8
    encodeNum2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 9
    inputDims = len(encodeNum1)

    # 初始化神经网络
    accumulatorNeuralNetwork = AccumulatorNeuralNetwork(inputDims)
    bridgeNeuralNetwork = BridgeNeuralNetwork(inputDims)
    complementNetwork = ComplementNetwork(inputDims)

    # 10 变为 190（补码操作）
    converseInput2 = complementNetwork.forward(encodeNum2)

    # 计算 encodeNum1 和 converseInput2 的结果
    accumulatorNeuralNetwork.compute(encodeNum1, converseInput2)

    # 打印输出
    print(accumulatorNeuralNetwork.outputs)
    mergedArray = accumulatorNeuralNetwork.outputs

    # 将 double 数组转换为 int 数组
    intResultArray = mergedArray.astype(int)

    # 模拟获取最终结果的索引
    finalResultIndex = bridgeNeuralNetwork.simulate(intResultArray)

    # 将索引还原为向量
    finalEncode = np.zeros(inputDims, dtype=int)
    resultDegree = 0
    for i in finalResultIndex:
        finalEncode[i] = 1
        resultDegree += 1

    # 打印最终结果
    print(resultDegree)

def main():
    start_time = time.time_ns()

    # Initialize HeadDirectionNeuralNetwork
    ring = HeadDirectionNeuralNetwork(360, 100, 1.0, 1.0, 3, 4)

    # Simulate with input angles [265, 266, 267, 268, 269]
    encode = ring.simulate([65, 66, 67, 68, 69])
    encode_vector = [0] * len(encode[-1])
    threshold_value = 0
    for i, val in enumerate(encode[-1]):
        if val:  # If True, set corresponding vector element to 1
            encode_vector[i] = 1
            threshold_value += 1

    # Initialize and simulate ReturnToOriginNeuralNetwork
    ellipsoid = ReturnToOriginNeuralNetwork(360, threshold_value)
    result_state = ellipsoid.simulate(encode_vector)

    # Reference output for 45 degrees
    result_state_reference_to_out = [1] * 45 + [0] * 315

    # Initialize and compute with AccumulatorNeuralNetwork
    neural_network = AccumulatorNeuralNetwork(len(result_state))
    neural_network.compute(result_state, result_state_reference_to_out)

    print(f"{list(neural_network.outputs)}\n")
    merged_array = neural_network.outputs

    # Convert double array to int array
    int_result_array = [int(x) for x in merged_array]

    # Initialize and simulate BridgeNeuralNetwork
    bridge_neural_network = BridgeNeuralNetwork(360)
    final_result_index = bridge_neural_network.simulate(int_result_array)

    # Convert indices to final encode vector
    final_encode = [0] * 360
    result_degree = 0
    for i in final_result_index:
        final_encode[final_result_index[i]] = 1
        result_degree += 1

    end_time = time.time_ns()
    duration = (end_time - start_time) / 1_000_000.0  # Convert to milliseconds
    print(f"运行时间（毫秒）: {duration}")
    print(result_degree)


if __name__ == "__main__":
    main_new()