import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Load the JSON data
file_path = r"C:\Users\rajes\Downloads\VsAda\del2.0\src\visual_output\2x2.json"
  # Adjust if needed
with open(file_path, "r") as f:
    data_json = json.load(f)

# Extract data points and labels
data = np.array(data_json["data"])  # shape: (N, 2)
labels = np.array(data_json["labels"])  # shape: (N,)

# Generate a mesh grid for decision boundary
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

# Create a dense grid of points
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),  # 300x300 grid
    np.linspace(y_min, y_max, 300)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]  # Reshape into a 2D array

# Use KDTree for fast nearest-neighbor lookup (approximate decision boundary)
tree = KDTree(data)
_, nearest_indices = tree.query(grid_points)
Z = labels[nearest_indices]  # Assign labels based on nearest real data point
Z = Z.reshape(xx.shape)  # Reshape back to grid format

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap="rainbow", levels=np.unique(Z))

# Overlay the actual points
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="rainbow", edgecolor="k")
plt.colorbar(scatter, ticks=np.unique(labels))  # Show only integer class labels

# Labels and Title
plt.title("Decision Boundary of Ada Model (JSON Output)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

# Show the plot
plt.show()
