import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load JSON file
with open("model_output.json", "r") as f:
    data = json.load(f)

X = np.array(data["data"])  # Input features (X, Y)
y = np.array(data["labels"])  # Ground truth labels
predictions = np.array(data["predictions"])  # Model predictions (confidence scores)

# Convert softmax probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Define class colors
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFD700"])
cmap_bold = ["r", "g", "b", "gold"]

# Create a mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Create a grid of points and predict their class
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = np.random.choice([0, 1, 2, 3], size=grid_points.shape[0])  # Replace with model.predict(grid_points)

# Reshape grid predictions
Z = grid_predictions.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Plot training points
for i, color in enumerate(cmap_bold):
    plt.scatter(X[y == i, 0], X[y == i, 1], c=color, edgecolor="k", label=f"Class {i+1}")

plt.xlabel("Feature 1 (X)")
plt.ylabel("Feature 2 (Y)")
plt.title("Decision Boundary & Training Data")
plt.legend()
plt.show()