import os
import sys
import json
import time

import numpy as np


# helper function mimicking typescript's null-coalesce (??) operator
def NullCoalesce(left, right):
    return left if left is not None else right


# helper function to write generated data to a json file
def WriteData(data: list, labels: list):
    # format generated data and output it to a file
    print("Writing generated data...")
    generatedData = {
        "data": data,
        "labels": labels
    }
    timestamp = time.time()
    os.makedirs("data", exist_ok=True)
    with open(f"data/{timestamp}.json", 'w', encoding='utf-8') as file:
        json.dump(generatedData, file, ensure_ascii=False, indent=4)
    print("Done writting generated data!")
    print("Ending program...")


# generates and labels data in a grid pattern
def GenerateGridData():
    # base values used to store the data passed to json file
    data = []
    labels = []

    # generate data
    print('Generating data...')
    for q in range(quantity):
        values = []
        for d in range(dimension):
            num = np.random.randint(lower, upper)
            values.append(num)
        data.append(values)
    
    # generate labels
    if (createLabels):
        print('Generating labels...')
        cell_size = (upper - lower) / types
        for point in data:
            grid_coords = [min(int((coord - lower) // cell_size), types - 1) for coord in point]
            label = sum(coord * (types ** i) for i, coord in enumerate(reversed(grid_coords))) + 1
            labels.append(label)
        print("Done generating!")
        
    # write data to file
    WriteData(data, labels)


# generates and labels data in a spiral pattern
def GenerateSpiralData():
    # base values used to store the data passed to json file
    data = []
    labels = []

    # Generate points for each spiral arm
    for i in range(types):
        print(f'Generating arm {i}...')
        # Generate radius values that increase along the spiral
        r = np.linspace(lower, upper, quantity//types)
        
        # Generate angle values with increasing radius
        # Add 2π/n_classes offset for each class to space arms evenly
        t = np.linspace(0, 3, quantity//types) + i * 2*np.pi/types
        
        # Add some noise
        t = t + noise * np.random.uniform(-1, 1, size=t.shape)
    
        # Convert polar coordinates to cartesian
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        x = x.tolist()
        y = y.tolist()
        
        # store data and labels
        for p in range(len(x)):     # p = point
            data.append([x[p], y[p]])
            labels.append(i + 1)
        print(f'Done generating arm {i}.')
        
    # write data to file
    WriteData(data, labels)


# ===== MAIN =====

# gets data from terminal, if available with a max of 5
args = [None] * 8
argsLength = len(sys.argv) - 1 if len(sys.argv) < 9 else 8
for i in range(argsLength):
    args[i] = sys.argv[i+1]


# integer indicating the size of the random dataset to be generated, defaults to 100
quantity: int = NullCoalesce(int(args[0]), 100)
# integer indicating labels that can be given to the data, defaults to 4
types: int = NullCoalesce(int(args[1]), 4)
# integer indicating the dimension of the points in the random dataset, defaults to 2
dimension: int = NullCoalesce(int(args[2]), 2)
# integer indicating lower range for the generated values, defaults to -100
lower: int = NullCoalesce(int(args[3]), -100)
# integer indicating upper range for the generated values, defaults to 100
upper: int = NullCoalesce(int(args[4]), 100)
# integer indicating shape of data to generate:
#   1: grid
#   2: spiral
shape: int = NullCoalesce(int(args[5]), 1)
# integer indicating if labels should be generated, 1 to generate and any other value to not
createLabels: bool = (NullCoalesce(int(args[6]), 1) == 1)
# decimal indicating noise multiplier if spiral generation is chosen
noise: float = NullCoalesce(float(args[7]), 0.1)

print(f"-=~=- Values -=~=-\nQuantity: {quantity}\nTypes: {types}\nDimension: {dimension}\nLower: {lower}\nUpper: {upper}\nShape: {shape}\nCreate Labels: {createLabels}\n\n")


# basic check to make sure passed parameters won't be an issue
quantity = int(quantity)
types = int(types)
dimension = int(dimension)
lower = int(lower)
upper = int(upper)
shape = int(shape)
if lower > upper:
    raise ValueError('Lower bound (4th passed value) cannot be greater than upper bound (5th passed value)!')
if types < 1:
    raise ValueError('There must be at least 1 type for the data to be assigned to!')


# attempt to generate data in requested shape
if shape == 1:
    GenerateGridData()
elif shape == 2:
    if (dimension == 2):
        GenerateSpiralData()
    else:
        print('Spiral generation only works for 2D!')
else:
    raise ValueError(f'Shape value "{shape}" is not a valid option!')
