import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("model_output.json") as f:
    data = json.load(f)

X = np.array(data["data"])
y_true = np.array(data["labels"])
y_pred = np.argmax(np.array(data["predictions"]), axis=1)

# Scatter plot of predictions
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis", alpha=0.8, s=40, edgecolor='k')
plt.title("Model Predictions (by Softmax Argmax)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.colorbar(scatter, label="Predicted Class")
plt.show()
