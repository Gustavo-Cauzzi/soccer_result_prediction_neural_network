import csv
import random
import math
import json
import time

class Neuron:
    def __init__(self):
        self.buffer = []
        self.value = 0 
        self.erro = 0

    def __str__(self) -> str:
        return f"Neuron v = {self.value}\n"

data = []
y = []
test_data_ratio = 0.2 # 10% dos dados serão separados para teste
learning_ratio = 0.3    
MAX_PATIANCE = 5

result_row = 'FTR'
unused_headers = ['FTHG', 'FTAG', 'HTR', 'HomeTeam', 'AwayTeam', 'ID'] 

with open("./Dataset_futebol.csv", 'r') as file:
    reader = csv.reader(file, delimiter=";")
    headers = next(reader)

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
                pass
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

def generate_test_data(x):
    classes_occurrences = {}
    for result in y:
        if result in classes_occurrences:
            classes_occurrences[result] += 1
        else:
            classes_occurrences[result] = 1
    
    for key, value in classes_occurrences.items():
        classes_occurrences[key] = value * test_data_ratio

    new_x = []
    test_x = []

    new_y = []
    test_y = []

    for idx, row in enumerate(x):
        if classes_occurrences[y[idx]] > 1: # Ainda temos que pegar classes que são desse resultado
            test_x.append(row)
            test_y.append(y[idx])
            classes_occurrences[y[idx]] -= 1
        else:
            new_x.append(row)
            new_y.append(y[idx])

    print(len(new_x), len(test_x))
    return new_x, test_x, new_y, test_y

data, test_data, y, test_y = generate_test_data(data)

print(len(data), len(test_data))

num_headers = len(headers)
last_layer = 3
layers_size = [num_headers, (num_headers + last_layer) // 2, last_layer]
layers_indexes_ranges = []  

print(layers_size)

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
    [random.random() for _ in range(neurons_qtd)] for _ in range(neurons_qtd)
]

def sigmoidal(value):
    return 1 / (1 + math.exp(-value))

result_map = {
    'H': [1, 0, 0],
    'A': [0, 1, 0],
    'D': [0, 0, 1],
}

def exec_neural_network_for_row(row):
    for header_index, header in enumerate(headers):
        neurons[header_index].value = row[header]

    for layer_index in range(len(layers_size)):
        if layer_index != 0:
            for neuron_index in range(layers_indexes_ranges[layer_index - 1], layers_indexes_ranges[layer_index]):
                neuron = neurons[neuron_index]
                neuron.value = sigmoidal(sum(neuron.buffer))
                neuron.buffer = []

        if layer_index == len(layers_size) - 1:
            break

        initial_index = 0 if layer_index == 0 else layers_indexes_ranges[layer_index - 1]
        for neuron_index in range(initial_index, layers_indexes_ranges[layer_index]):
            current_neuron_value = neurons[neuron_index].value
            for next_layer_neuron_index in range(layers_indexes_ranges[layer_index], layers_indexes_ranges[layer_index + 1]):
                neurons[next_layer_neuron_index].buffer.append(current_neuron_value * weights_matrix[neuron_index][next_layer_neuron_index])

    results = []
    for neuron_index in range(layers_indexes_ranges[1], layers_indexes_ranges[2]):
        results.append(neurons[neuron_index].value)
    return results

last_epoch_error = 0
current_patiance = 0
best_wheights_matrix = weights_matrix
while True: # Loop de épocas
    current_epoch_error = 0

    for row_index, row in enumerate(data):
        exec_neural_network_for_row(row)

        expected_results = result_map[y[row_index]]
        for idx, neuron_index in enumerate(range(layers_indexes_ranges[1], layers_indexes_ranges[2])):
            neuron = neurons[neuron_index]
            error_fator = (expected_results[idx] - neuron.value)
            neuron.error = neuron.value * (1 - neuron.value) * error_fator
            current_epoch_error += abs(neuron.error)

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
                    weights_matrix[neuron_index][next_layer_neuron_index] = \
                        weights_matrix[neuron_index][next_layer_neuron_index] + learning_ratio * \
                        current_neuron_value * next_layer_neuron_error
    
    if last_epoch_error - 0.4 < current_epoch_error:
        current_patiance += 1
        print(f"-------------------- ÉPOCA PIORRRRRR --------------------- {last_epoch_error} {current_epoch_error} ", current_patiance)
    else:
        best_wheights_matrix = [row.copy() for row in weights_matrix]
        print(f"-=-=-=-=- ÉPOCA MELHOR -=-=-=-=-- {last_epoch_error} {current_epoch_error}" )
        current_patiance = 0
    
    last_epoch_error = current_epoch_error
    if current_patiance > MAX_PATIANCE:
        break

weights_matrix = best_wheights_matrix

confusion_matrix = {
    "H": { "H": 0, "D": 0, "A": 0 },
    "A": { "H": 0, "D": 0, "A": 0 },
    "D": { "H": 0, "D": 0, "A": 0 },
}
acertou = 0
errou = 0
for test_row_index, test_row in enumerate(test_data):
    results = exec_neural_network_for_row(test_row)
    max_neuron_index = results.index(max(results))

    calculated_result = list(result_map.keys())[max_neuron_index]
    expected_result = test_y[test_row_index]
    if calculated_result == expected_result:
        acertou += 1
    else:
        errou += 1

    confusion_matrix[expected_result][calculated_result] += 1

    print("TEST ROW ----- ")
    print(f'results: {results}')
    print(f'max_neuron_index: {max_neuron_index}')
    print(f'calculated_result: {calculated_result}')
    print(f'test_y[test_row_index]: {test_y[test_row_index]}')

log_buffer = ''
def log(s):
    global log_buffer 
    log_buffer += s + '\n'
    print(s)

log(f'used_headers {headers}')

log(f'Erro final considerado: {last_epoch_error}')

log("Confusion Matrix:")
for key, values in confusion_matrix.items():
    log(f'{key} | {values}')

for key, values in confusion_matrix.items():
    log(f"Parâmetros do resultado:  {key}")
    vps = confusion_matrix[key][key]
    fns_vps = sum(confusion_matrix[key].values())
    fps_vps = sum(map(lambda values: values[key], confusion_matrix.values()))

    recall = vps / fns_vps
    precision = vps / fps_vps
    f1 = 2 * precision * recall / (precision + recall)
    log(f"Recall: {recall} ({vps}, {fns_vps})")
    log(f'Precision: {precision} ({vps}, {fps_vps})')
    log(f'F1-score: {f1} ({vps}, {fps_vps})')

log(f'acertou: {acertou}')
log(f'errou: {errou}')
log(f'Acurácia: {acertou * 100 / len(test_data)} {len(test_data)}')

with open(f"Output{time.time()}.txt", "w") as text_file:
    text_file.write(log_buffer)
    text_file.write(json.dumps(weights_matrix))