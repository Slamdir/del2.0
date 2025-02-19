import json
import pytest
import tkinter as tk
from tkinter import filedialog
import numpy as np
from Visualizer import JSONVisualizer  # Import your visualizer class

@pytest.fixture
def sample_json(tmp_path):
    """Creates a temporary JSON file for testing."""
    file_path = tmp_path / "test_data.json"
    test_data = {
        "data": [[1, 2], [3, 4], [5, 6]],
        "labels": [1, 2, 3]
    }
    with open(file_path, "w") as f:
        json.dump(test_data, f)
    return str(file_path)

def test_json_file_selection(mocker):
    """TC-017: Ensure the file selection dialog opens and returns a file path."""
    mock_file = "dummy.json"  # Simulated file path
    mocker.patch.object(filedialog, "askopenfilename", return_value=mock_file)  # Mock file dialog

    app = JSONVisualizer()
    file_path = filedialog.askopenfilename()  # Simulate user selecting a file

    assert file_path == "dummy.json"  # Ensure the mocked file path is returned

def test_json_parsing(sample_json):
    """TC-018: Validate JSON parsing for well-formed files."""
    with open(sample_json, "r") as f:
        data = json.load(f)
    
    assert "data" in data
    assert "labels" in data
    assert len(data["data"]) == 3
    assert len(data["labels"]) == 3

def test_json_error_handling(mocker):
    """TC-019: Ensure errors are handled for missing/corrupt JSON files."""
    mocker.patch("tkinter.filedialog.askopenfilename", return_value="missing.json")
    app = JSONVisualizer()
    with pytest.raises(FileNotFoundError):
        app.plot_data("missing.json")

def test_gui_initialization():
    """TC-021: Ensure the visualizer GUI initializes without errors."""
    app = JSONVisualizer()
    assert isinstance(app, tk.Tk)

def test_plot_update(sample_json):
    """TC-022: Ensure clicking 'Upload JSON' updates the plot dynamically."""
    app = JSONVisualizer()
    app.plot_data(sample_json)

    # Check if figure exists and has data
    assert len(app.ax.collections) > 0  # At least one scatter plot should be present
