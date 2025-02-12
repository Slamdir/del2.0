import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

# Load JSON Data
#file_path = r"C:\Users\rajes\Downloads\VsAda\del2.0\src\visual_output\150x3_spiral_data.json"
file_path = r"C:\Users\rajes\Downloads\VsAda\del2.0\src\visual_output\1737997510.2495284.json"
with open(file_path, "r") as f:
    data_json = json.load(f)

# Extract data and labels
data = np.array(data_json["data"])   # (N,2)
labels = np.array(data_json["labels"])  # (N,)

# Normalize labels (Convert to float for interpolation)
labels = labels.astype(float)

# Generate a mesh grid for decision boundary
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),  # 300x300 grid
    np.linspace(y_min, y_max, 300)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten to (N,2)

# Apply RBF Interpolation
rbf = RBFInterpolator(data, labels, kernel="thin_plate_spline")  # Try other kernels if needed
Z = rbf(grid_points).reshape(xx.shape)  # Reshape back to grid format

# Plot Decision Boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap="rainbow", levels=np.unique(labels))

# Overlay Data Points
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="rainbow", edgecolor="k")
plt.colorbar(scatter, ticks=np.unique(labels))  # Show only integer labels

# Labels & Title
plt.title("Decision Boundary using RBF Interpolation")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

# Show Plot
plt.show()
