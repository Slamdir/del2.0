import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class JSONVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Ada Model JSON Visualizer")
        self.geometry("800x600")

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_title("Ada Model Data Visualization")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        # Embed Matplotlib in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Upload Button
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

        # Extract (x, y) coordinates and labels
        data = np.array(data_json["data"])  # shape: (N, 2)
        labels = np.array(data_json["labels"])  # shape: (N,)

        # Clear previous plot
        self.ax.clear()
        self.ax.set_title("Ada Model Data Visualization")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        # Scatter Plot
        scatter = self.ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="rainbow", edgecolor="k")
        self.fig.colorbar(scatter, ax=self.ax, ticks=np.unique(labels))

        # Update plot
        self.canvas.draw()

# Run the GUI
if __name__ == "__main__":
    app = JSONVisualizer()
    app.mainloop()
