import csv
import random
import sys
import math


class Layer:
    def __init__(neurons):
        self.neurons = neurons

class Neuron:
    def __init__(self):
        self.buffer = []
        self.value = 0 
        self.erro = 0

    def __str__(self) -> str:
        return f"Neuron v = {self.value}\n"

data = []
y = []
result_row = 'FTR'
unused_headers = ['FTHG', 'FTAG','HTR', 'HF', 'AF', 'HY', 'AY', 'HomeTeam', 'AwayTeam', 'HS', 'AS', 'ID'] # Talvez tirar os HALF TIME

# Lembrar que é necessário adicionar o "ID" dentro do csv

# X

with open("./Dataset_futebol.csv", 'r') as file:
    reader = csv.reader(file, delimiter=";")
    headers = next(reader) # Read the header row

    for row in reader:
        record = {}
        for i, value in enumerate(row):
            record[headers[i]] = value
        for header in unused_headers:
            record.pop(header)
        y.append(record[result_row])
        record.pop(result_row)
        for key in record.keys():
            try:
                record[key] = int(record[key])
            except:
                a = 1
        data.append(record)
    
    headers = list(filter(lambda h: not h in unused_headers and h != result_row, headers))
    for key in headers:
        if key == result_row:
            continue
        biggest = 0
        lowest = float('inf')
        for row in data:
            if row[key] < lowest:
                lowest = row[key]
            if row[key] > biggest:
                biggest = row[key]
        for row in data:
            row[key] = (row[key] - lowest) / (biggest - lowest)

num_headers = len(headers)
last_layer = 3
layers_size = [num_headers, (num_headers + last_layer) // 2, last_layer]
layers_indexes_ranges = []  

learning_ratio = 0.2571649

s = 0
for size in layers_size:
    s += size
    layers_indexes_ranges.append(s)

# print(data)
neurons = [
    Neuron() for _ in range(sum(layers_size))
]

neurons_qtd = sum(layers_size)
weights_matrix = [
    [random.random() for i in range(neurons_qtd)] for j in range(neurons_qtd)
]

# print(weights_matrix)
# print(y)
# print('neurons_qtd: ', layers_size)

def sigmoidal(value):
    return 1 / (1 + math.exp(-value))

# print(headers)

result_map = {
    'H': [1, 0, 0],
    'A': [0, 1, 0],
    'D': [0, 0, 1],
}

for row_index, row in enumerate(data):
    for header_index, header in enumerate(headers):
        neurons[header_index].value = row[header]

    for layer_index in range(len(layers_size)):
        if layer_index != 0:
            for neuron_index in range(layers_indexes_ranges[layer_index - 1], layers_indexes_ranges[layer_index]):
                neuron = neurons[neuron_index]
                neuron.value = sigmoidal(sum(neuron.buffer))

        if layer_index == len(layers_size) - 1:
            break

        initial_index = 0 if layer_index == 0 else layers_indexes_ranges[layer_index - 1]
        for neuron_index in range(initial_index, layers_indexes_ranges[layer_index]):
            current_neuron_value = neurons[neuron_index].value
            for next_layer_neuron_index in range(layers_indexes_ranges[layer_index], layers_indexes_ranges[layer_index + 1]):
                neurons[next_layer_neuron_index].buffer.append(current_neuron_value * weights_matrix[neuron_index][next_layer_neuron_index])
        
    expected_results = result_map[y[row_index]]
    for idx, neuron_index in enumerate(range(layers_indexes_ranges[1], layers_indexes_ranges[2])):
        neuron = neurons[neuron_index]
        error_fator = (expected_results[idx] - neuron.value)
        neuron.error = neuron.value * (1 - neuron.value) * error_fator

    for neuron_index in range(layers_indexes_ranges[0], layers_indexes_ranges[1]):
        neuron = neurons[neuron_index]
        error_fator = 0
        for last_layer_neuron_index in range(layers_indexes_ranges[1], layers_indexes_ranges[2]):
            error_fator += neurons[last_layer_neuron_index].error * weights_matrix[neuron_index][last_layer_neuron_index]
        neuron.error = neuron.value * (1 - neuron.value) * error_fator

    for layer_index in range(len(layers_size) - 1):
        initial_index = 0 if layer_index == 0 else layers_indexes_ranges[layer_index - 1]
        for neuron_index in range(initial_index, layers_indexes_ranges[layer_index]):
            current_neuron_value = neurons[neuron_index].value
            for next_layer_neuron_index in range(layers_indexes_ranges[layer_index], layers_indexes_ranges[layer_index + 1]):
                next_layer_neuron_error = neurons[next_layer_neuron_index].error
                weights_matrix[neuron_index][next_layer_neuron_index] = weights_matrix[neuron_index][next_layer_neuron_index] + learning_ratio * current_neuron_value * next_layer_neuron_error
              
print(weights_matrix)