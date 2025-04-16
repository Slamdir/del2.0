import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        self.colorbar = None

        self.data = None
        self.labels = None
        self.grid = None
        self.current_file = None

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

        self.data = np.array(data_json["data"])

        if "labels" in data_json:
            self.labels = np.array(data_json["labels"])
        else:
            self.labels = np.zeros(self.data.shape[0], dtype=int)

        if "grid" in data_json:
            self.grid = np.array(data_json["grid"])
        else:
            self.grid = None

        # Clear previous plot
        self.ax.clear()

        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        # Calculate dynamic padding 
        x_padding = (self.data[:, 0].max() - self.data[:, 0].min()) * 0.002
        y_padding = (self.data[:, 1].max() - self.data[:, 1].min()) * 0.002

        self.x_min = self.data[:, 0].min() - x_padding
        self.x_max = self.data[:, 0].max() + x_padding
        self.y_min = self.data[:, 1].min() - y_padding
        self.y_max = self.data[:, 1].max() + y_padding

        self.ax.set_title("Ada Model Predicted Labels with Decision Boundaries")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        # Plot decision boundary
        if self.grid is not None:
            self.plot_decision_boundary_from_grid()

        # Scatter training data
        scatter = self.ax.scatter(
            self.data[:, 0], self.data[:, 1],
            c=self.labels,
            cmap="viridis",
            edgecolor="k",
            s=80
        )

        if np.unique(self.labels).size > 1:
            self.colorbar = self.fig.colorbar(scatter, ax=self.ax, ticks=np.unique(self.labels))

        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)

        self.canvas.draw_idle()

    def plot_decision_boundary_from_grid(self):
        if self.grid is None:
            return
        
        grid_points = self.grid[:, :2]
        grid_labels = self.grid[:, 2].astype(int)

        x_unique = np.sort(np.unique(grid_points[:, 0]))
        y_unique = np.sort(np.unique(grid_points[:, 1]))
        xx, yy  = np.meshgrid(x_unique, y_unique)

        # reshape in Fortran order to match Ada’s X‑outer, Y‑inner loop
        label_map = grid_labels.reshape(yy.shape, order='F')

        base = plt.get_cmap("viridis")
        colors = base(np.linspace(0, 1, 3))
        cmap = ListedColormap(colors)

        # set up bin edges so that class=1→bin [0.5,1.5), 2→[1.5,2.5), 3→[2.5,3.5)
        levels = [0.5, 1.5, 2.5, 3.5]
        norm   = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        self.ax.contourf(
            xx, yy, label_map,
            levels=levels,
            cmap=cmap,
            norm=norm,
            alpha=0.3,
            extend="neither"
        )

        # draw the actual boundaries as lines
        self.ax.contour(
            xx, yy, label_map,
            levels=[1.5, 2.5],
            colors='k',
            linewidths=1
        )

    def save_plot(self):
        if self.current_file is None:
            return 

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

if __name__ == "__main__":
    app = JSONVisualizer()
    app.mainloop()
