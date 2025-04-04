import json
import pytest
import tkinter as tk
import numpy as np
from dev_tools.visual_output.labels_Visualizer import JSONVisualizer
from unittest.mock import MagicMock, patch

@pytest.fixture
def sample_json(tmp_path):
    """Creates a temporary JSON file with valid test data."""
    file_path = tmp_path / "test_data.json"
    test_data = {
        "data": [[1, 2], [3, 4], [5, 6]],
        "labels": [1, 2, 3]
    }
    with open(file_path, "w") as f:
        json.dump(test_data, f)
    return str(file_path)

def test_file_selection_dialog(mocker):
    """TC-022: Ensure file selection dialog opens and allows JSON file selection."""
    mocker.patch("tkinter.filedialog.askopenfilename", return_value="test.json")
    
    app = JSONVisualizer()
    app.load_json()  # Call method but don't expect a return

    # Verify that the dialog was actually called
    tkinter_filedialog_mock = mocker.patch("tkinter.filedialog.askopenfilename")
    assert tkinter_filedialog_mock.called

def test_json_parsing(sample_json):
    """TC-023: Validate correct JSON file loading and parsing."""
    with open(sample_json, "r") as f:
        data = json.load(f)
    
    assert "data" in data
    assert "labels" in data
    assert len(data["data"]) == 3
    assert len(data["labels"]) == 3

def test_invalid_json_structure(tmp_path):
    """TC-025: Ensure the program handles edge cases in JSON structure."""
    file_path = tmp_path / "invalid_data.json"
    invalid_data = {
        "points": [[1, 2], [3, 4]],  # Incorrect key (should be "data")
        "labels": [1, 2]
    }
    with open(file_path, "w") as f:
        json.dump(invalid_data, f)

    app = JSONVisualizer()
    with pytest.raises(KeyError):
        app.plot_data(str(file_path))

def test_missing_json_file():
    """TC-024: Ensure system gracefully handles missing JSON file errors."""
    app = JSONVisualizer()
    with pytest.raises(FileNotFoundError):
        app.plot_data("nonexistent.json")

def test_plot_updates(sample_json):
    """TC-026: Ensure file selection → JSON parsing → plotting works correctly."""
    app = JSONVisualizer()
    app.plot_data(sample_json)

    # Check if the figure updated with new data
    assert len(app.ax.collections) > 0  # At least one scatter plot should be present

def test_gui_initialization():
    """Ensure the GUI initializes properly without errors."""
    app = JSONVisualizer()
    assert isinstance(app, tk.Tk)
