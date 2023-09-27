import csv
import random
import sys

class Layer:
    def __init__(neurons):
        self.neurons = neurons

class Neuron:
    def __init__(weight):
        # aa
        print("bdiwua")

data = []
unused_headers = ['FTHG', 'FTAG','HTR', 'HF', 'AF', 'HY', 'AY', 'HomeTeam', 'AwayTeam', 'HS', 'AS'] # Talvez tirar os HALF TIME
result_row = 'FTR'

with open("./Dataset_futebol.csv", 'r') as file:
    reader = csv.reader(file, delimiter=";")
    headers = next(reader) # Read the header row

    for row in reader:
        record = {}
        for i, value in enumerate(row):
            record[headers[i]] = value
        for header in unused_headers:
            record.pop(header)
        for key in record.keys():
            try:
                record[key] = int(record[key])
            except:
                a = 1
        data.append(record)
    
    headers = list(filter(lambda h: not h in unused_headers, headers))
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
            row[key] = (row[key] - lowest) / biggest


num_headers = len(headers)
last_layer = 3
layers_size = [num_headers, (num_headers + last_layer) // 2, last_layer]

print(data)
neurons = [
    Neuron() for _ in range(sum(layers_size))
]

neurons_qtd = sum(layers_size)
print('neurons_qtd: ', neurons_qtd)
weights_matrix = [
    [random.random() for i in range(neurons_qtd)] for j in range(neurons_qtd)
]

print(weights_matrix)

