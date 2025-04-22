import argparse
import os
import json
import time
import numpy as np


# helper function to write generated data to a json file
def WriteData(data: list, labels: list):
    # format generated data and output it to a file
    print("Writing generated data...")
    generatedData = {
        "data": data,
        "labels": labels
    }
    timestamp = time.time()
    os.makedirs("generated-data", exist_ok=True)
    with open(f"generated-data/{timestamp}.json", 'w', encoding='utf-8') as file:
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

    # Calculate spiral parameters based on input bounds
    max_radius = (upper - lower) * 0.5 # helps keep radius consistent with upper and lower limits, best if upper = -lower
    max_angle = rotations * 2 * np.pi

    # Generate points for each spiral arm
    for i in range(types):
        print(f'Generating arm {i}...')
        
        # Number of points per spiral
        n_points = quantity // types
        
        # Generate angles with full rotations
        theta = np.linspace(0, max_angle, n_points)
        
        # Archimedean spiral formula (r = a + b*θ)
        r = (max_radius / theta[-1]) * theta
        
        # Add proportional noise to radius
        r += noise * np.random.randn(n_points) * (r * 0.1)
        
        # Convert to cartesian with class offset
        x = r * np.cos(theta + i * 2*np.pi/types)
        y = r * np.sin(theta + i * 2*np.pi/types)
        
        # Store data and labels
        for p in range(n_points):
            data.append([x[p], y[p]])
            if (createLabels): labels.append(i + 1)

        print(f'Done generating arm {i}.')
    
    # Write data to file
    WriteData(data, labels)



# ===== MAIN =====

parser = argparse.ArgumentParser(
    description='Data Pattern Generator',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
    
# Core parameters
parser.add_argument('-q', '--quantity', type=int, default=300,
                    help='Number of data points to generate')
parser.add_argument('-t', '--types', type=int, default=3,
                    help='Number of distinct classes/labels (to the power of types in case of grid generation)')
parser.add_argument('-d', '--dimension', type=int, default=2,
                    help='Dimensionality of data points (must be 2 for spiral generation)')
parser.add_argument('-l', '--lower', type=int, default=-1,
                    help='Lower bound for generated values')
parser.add_argument('-u', '--upper', type=int, default=1,
                    help='Upper bound for generated values')

# Generation options
parser.add_argument('-s', '--shape', type=int, default=2, choices=[1,2],
                    help='Data patterns:\n1=Grid\n2=Spiral')
parser.add_argument('-n', '--noise', type=float, default=0.5,
                    help='Noise level for spiral generation')
parser.add_argument('-r', '--rotations', type=float, default=0.5,
                    help='Number of spiral rotations')

# Flags
parser.add_argument('--no-labels', action='store_false', dest='create_labels',
                    help='Disable label generation')

args = parser.parse_args()

# Assign parsed arguments
quantity = args.quantity
types = args.types
dimension = args.dimension
lower = args.lower
upper = args.upper
shape = args.shape
createLabels = args.create_labels
noise = args.noise
rotations = args.rotations

# Validation checks remain unchanged
if lower > upper:
    raise ValueError('Lower bound cannot exceed upper bound')
if types < 1:
    raise ValueError('At least 1 type required')


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
