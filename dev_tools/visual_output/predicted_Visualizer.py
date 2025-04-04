import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class JSONVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Ada Model JSON Visualizer with Decision Boundaries")
        self.geometry("1000x800")

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("Ada Model Predicted Labels with Decision Boundaries")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        # Embed Matplotlib in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Button Frame
        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        load_button = tk.Button(button_frame, text="Upload JSON File", command=self.load_json)
        load_button.pack(side=tk.LEFT, padx=5, pady=5)

    def load_json(self):
        file_path = filedialog.askopenfilename(
            title="Open JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.plot_data(file_path)

    def plot_data(self, file_path):
        # Load JSON Data
        with open(file_path, "r") as f:
            data_json = json.load(f)

        # Extract (x, y) coordinates
        data = np.array(data_json["data"])  # shape: (N, 2)

        # Extract predicted labels directly
        predicted_labels = np.array(data_json["labels"])  # shape: (N,)

        # Clear previous plot
        self.ax.clear()
        self.ax.set_title("Ada Model Predicted Labels with Decision Boundaries")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        # Generate decision boundary
        self.plot_decision_boundary(data, predicted_labels)

        # Scatter Plot of actual points
        scatter = self.ax.scatter(
            data[:, 0], data[:, 1],
            c=predicted_labels,
            cmap="viridis",
            edgecolor="k",
            s=40
        )
        self.fig.colorbar(scatter, ax=self.ax, ticks=np.unique(predicted_labels))

        # Update plot
        self.canvas.draw()

    def plot_decision_boundary(self, data, predicted_labels):
        # Define a fine grid
        x_min, x_max = data[:, 0].min() - 5, data[:, 0].max() + 5
        y_min, y_max = data[:, 1].min() - 5, data[:, 1].max() + 5
        h = 0.2  # fine step size

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Flatten grid points
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Approximate prediction by nearest neighbor
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=15, weights="distance")
        knn.fit(data, predicted_labels)

        Z = knn.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        self.ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

# Run the GUI
if __name__ == "__main__":
    app = JSONVisualizer()
    app.mainloop()
