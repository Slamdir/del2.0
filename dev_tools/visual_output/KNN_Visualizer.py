import json 
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # No NavigationToolbar2Tk
import os

class JSONVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Ada Model JSON Visualizer with Decision Boundaries")
        self.geometry("1000x800")

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        load_button = tk.Button(button_frame, text="Upload JSON File", command=self.load_json)
        load_button.pack(side=tk.LEFT, padx=5, pady=5)

        save_button = tk.Button(button_frame, text="Save Plot", command=self.save_plot)
        save_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None

        self.colorbar = None   # <<==== Track the colorbar object!

    def load_json(self):
        file_path = filedialog.askopenfilename(
            title="Open JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.current_file = file_path
            self.plot_data(file_path)

    def plot_data(self, file_path):
        with open(file_path, "r") as f:
            data_json = json.load(f)

        data = np.array(data_json["data"])

        if "labels" in data_json:
            predicted_labels = np.array(data_json["labels"])
        else:
            predicted_labels = np.zeros(data.shape[0], dtype=int)

        # Clear previous plot
        self.ax.clear()

        # Remove old colorbar if it exists
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None  # << Important to reset pointer

        # Calculate dynamic padding again
        x_padding = (data[:, 0].max() - data[:, 0].min()) * 0.1
        y_padding = (data[:, 1].max() - data[:, 1].min()) * 0.1

        self.x_min = data[:, 0].min() - x_padding
        self.x_max = data[:, 0].max() + x_padding
        self.y_min = data[:, 1].min() - y_padding
        self.y_max = data[:, 1].max() + y_padding

        self.ax.set_title("Ada Model Predicted Labels with Decision Boundaries")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        # Plot decision boundary
        self.plot_decision_boundary(data, predicted_labels)

        scatter = self.ax.scatter(
            data[:, 0], data[:, 1],
            c=predicted_labels,
            cmap="viridis",
            edgecolor="k",
            s=80
        )

        if np.unique(predicted_labels).size > 1:
            self.colorbar = self.fig.colorbar(scatter, ax=self.ax, ticks=np.unique(predicted_labels))

        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)

        # Use draw_idle instead of draw
        self.canvas.draw_idle()

    def plot_decision_boundary(self, data, predicted_labels):
        range_x = self.x_max - self.x_min
        range_y = self.y_max - self.y_min

        h = max(range_x, range_y) / 300
        
        xx, yy = np.meshgrid(
            np.arange(self.x_min, self.x_max, h),
            np.arange(self.y_min, self.y_max, h)
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=12, weights="distance")
        knn.fit(data, predicted_labels)

        Z = knn.predict(grid_points)
        Z = Z.reshape(xx.shape)

        self.ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

    def save_plot(self):
        if self.current_file is None:
            return  # No file loaded yet, nothing to save!

        file_name = os.path.splitext(os.path.basename(self.current_file))[0]
        save_path = filedialog.asksaveasfilename(
            title="Save Plot As...",
            defaultextension=".png",
            initialfile=f"{file_name}_plot",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf")]
        )
        if save_path:
            self.fig.savefig(save_path, bbox_inches='tight')
            tk.messagebox.showinfo("Save Successful", f"Plot saved successfully:\n{save_path}")

# Run the GUI
if __name__ == "__main__":
    app = JSONVisualizer()
    app.mainloop()
