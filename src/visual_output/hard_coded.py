import tkinter as tk 
from tkinter import filedialog
import json
import matplotlib
matplotlib.use("TkAgg") 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class JSONDataPlotter(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("3-Region Decision Boundary")
        self.geometry("800x600")

        # the Matplotlib Figure and Axes
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Scatter Plot with 3-Region Boundary")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        
        # the Matplotlib figure in the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # frame for buttons
        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # button: load JSON
        load_button = tk.Button(button_frame, text="Load JSON", command=self.load_json)
        load_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # button: clear plot
        clear_button = tk.Button(button_frame, text="Clear Plot", command=self.clear_plot)
        clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # data storage
        self.data = None
        self.labels = None

    def load_json(self):
        # filepath
        file_path = filedialog.askopenfilename(
            title="Open JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            with open(file_path, "r") as f:
                json_data = json.load(f)
            
            # numpy arrays for slicing
            self.data = np.array(json_data["data"])   # shape: (N, 2)
            self.labels = np.array(json_data["labels"])  # shape: (N,)

            self.plot_data()

    def plot_data(self):
        # clear 
        self.ax.clear()

        self.ax.set_title("Scatter Plot with 3-Region Boundary")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        
        if self.data is not None and self.labels is not None:
            
            # range for X and Y
            x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
            y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
            
            # mesh grid
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 300),
                np.linspace(y_min, y_max, 300)
            )
            
            # shape (N, 2) to assign "region labels"
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            def swirl_region(x, y):
                angle = np.arctan2(y, x)  # -pi .. pi
                if angle < 0:
                    angle += 2*np.pi      # now 0..2*pi
                if angle < 2*np.pi/3:
                    return 0
                elif angle < 4*np.pi/3:
                    return 1
                else:
                    return 2
            
            # region labels to each point in the grid
            region_labels = []
            for (gx, gy) in grid_points:
                region_labels.append(swirl_region(gx, gy))
            
            region_labels = np.array(region_labels).reshape(xx.shape)
            
            # integer region labels
            self.ax.contourf(
                xx, yy, region_labels,
                alpha=0.4,
                cmap="rainbow",
                levels=[-0.5, 0.5, 1.5, 2.5]  # boundaries for 0,1,2
            )
            
            # plot the actual data points on top
            scatter = self.ax.scatter(
                self.data[:, 0], 
                self.data[:, 1],
                c=self.labels,
                cmap="rainbow",  
                alpha=0.9,
                edgecolor="k"
            )  
            
            # color bar on right
            cbar = self.fig.colorbar(scatter, ax=self.ax)
            cbar.set_label("Data Label")
            
            # integer labels
            unique_labels = np.unique(self.labels)
            cbar.set_ticks(unique_labels)
            cbar.set_ticklabels(unique_labels)
            
            # legend
            self.ax.text(0.05, 0.95, "3-region swirl boundary (hard-coded)",
                         transform=self.ax.transAxes, fontsize=9,
                         va='top', bbox=dict(boxstyle="round", fc="w", alpha=0.5))

        self.canvas.draw()

    def clear_plot(self):
        """Remove all data from the plot and redraw."""
        self.ax.clear()
        self.ax.set_title("Scatter Plot with 3-Region Boundary")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()

if __name__ == "__main__":
    app = JSONDataPlotter()
    app.mainloop()
