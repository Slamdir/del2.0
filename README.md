# DEL – Deep‑Learning Library for Ada  
*Team 19 Senior Design Project*

> **Mission** – deliver first‑class, fully type‑safe deep‑learning primitives to the Ada ecosystem.  

---

## 📑 Table of Contents
1. [Why DEL?](#-why-del)
2. [Prerequisites](#-prerequisites)
3. [Features](#-features)
4. [Quick Start](#-quick-start)
5. [Repository Layout](#-repository-layout)
6. [Framework Architecture](#-framework-architecture)
7. [Build & Install](#-build--install)
8. [Demos](#-demos)
9. [Data Management](#-data-management)
10. [Model Management](#-model-management)
11. [Export & Interoperate](#-export--interoperate)
---

## 🚀 Why DEL?
Ada powers mission‑critical software (avionics, rail, defence, medical) yet lacks a native deep‑learning stack.  
Python‑based frameworks are powerful but introduce runtime GC, certification hurdles, and large footprints.

---

## 🔧 Prerequisites
| Component | Purpose | Source |
|-----------|---------|--------|
| **Alire Package Manager** | Dependency & build manager | [Alire](https://alire.ada.dev/) |
| **Python 3.9+** *(optional)* | Runs demo/visual‑tools | [python.org](https://python.org) |
| **VS Code + Ada Extension** *(optional)* | IDE support | [VS Code](https://code.visualstudio.com/) |

**System requirements**  – Windows / Linux / macOS • ≥ 4 GB RAM (8 GB recommended) • ≥ 500 MB disk space.

> Install GNAT first, then run `alr get alire && alr setup` and verify with `alr --version`.

---

# ✨ Del Library Features

| Area | Module(s) | Highlights |
|------|-----------|------------|
| Tensor Operations | `del.ads/adb` | Matrix operations, tensor manipulation, mathematical functions |
| Neural Layers | `del-operators.*` | Linear, ReLU, Sigmoid, SoftMax, HyperTanh with forward/backward propagation |
| Loss Functions | `del-loss.*` | Cross-Entropy, Mean Square Error, Mean Absolute Error |
| Optimization | `del-optimizers.*` | SGD with momentum and weight decay support |
| Model Management | `del-model.*` | Layer composition, training loop, inference pipeline |
| Data Handling | `del-data.*` | Dataset creation, loading, and manipulation |
| File Formats | `del-json.*`, `del-yaml.*` | Data import from JSON and YAML formats |
| Export Capabilities | `del-export.*` | Model state persistence with visualization grid |
| ONNX Support | `del-onnx.*` | Neural network interchange format import/export |
| Utilities | `del-utilities.*` | Dataset shuffling, accuracy metrics, array helpers |

---

## 🚀 Quick Start
```bash
# Clone & build
git clone https://github.com/Slamdir/del2.0.git
cd del2.0
alr build            
```

---

## 🗂 Repository Layout
```
src/
  del.ads / del.adb         -- tensor core
  del-operators.*           -- algebra kernels
  del-model.*               -- model container & layers
  del-loss.*                -- loss functions
  del-optimizers.*          -- optimizers
  del-utilities.*           -- helpers (init, logging…)
  del-data.*                -- dataset helpers
  del-json.* / del-yaml.*   -- parsers
  del-export.*              -- flat‑file exporter
  del-onnx.*                -- ONNX emitter
demos/                      -- demo programs
del.gpr                     -- GNAT project file
```

---

## 🏗 Framework Architecture
| Module | Description |
|--------|-------------|
| **del-core** | Core types & initialization (root package) |
| **del-model** | Neural‑network container & layer registry |
| **del-operators** | Activation & linear algebra kernels |
| **del-loss** | Loss‑function implementations |
| **del-optimizers** | Optimisation algorithms |
| **del-data** | Data loading & batching helpers |
| **del-json / del-yaml** | Lightweight parsers for datasets & configs |
| **del-onnx** | ONNX exporter |
| **del-export** | Flat JSON weight exporter |
| **del-utilities** | Misc helpers (initialisers, logging, CLI) |

---

## 🛠 Build & Install
### With **Alire** *(recommended)*
```bash
alr build           # static build
# optional system‑wide install
```
## 🎓 Demos
```bash
# Generate synthetic classification data
python3 dev_tools/generator.py
```
---

## 📂 Data Management
DEL ingests **JSON** & **YAML** datasets and experiment configs.

```yaml
# sample YAML dataset
data:
  - [-86, -52]
  - [-100, -89]
labels:
  - 2
  - 4
```

```ada
-- Load data from file with automatic format detection
Model.Load_Data_From_File(
   Filename     => "data.yaml",
   Data_Shape   => (1, 2),
   Target_Shape => (1, 4));

-- Export model with predictions after training
Model.Export_To_JSON("output.json");
```

The library includes a `Combine_Dataset_Samples` function that can merge multiple training samples into a consolidated dataset, which is useful for creating batched training data from individual examples.

---

## 🧰 Dev Tools

### ⚙️ Data Generator

To use this data generation tool, developers can run the Python script with command-line arguments to create 2D/ND grid patterns or 2D spiral patterns. The tool generates JSON files with coordinate data and optional labels, which are stored in a timestamped file within the project directory.

**Quick Start**
- View all parameters: `python generator.py --help`
- Generate a default spiral dataset: `python generator.py -s 2 -q 1000`

**Key Options**
| Parameter | Description | Default |
|---------|---------|---------|
| `-s`/`--shape` | `1`=Grid, `2`=Spiral | 2 |
| `-q`/`--quantity` | Total data points | 300 |
| `-d`/`--dimension` | Data dimensions (2 for spirals) | 2 |
| `-t`/`--types` | Classes (grid) or spiral arms | 3 |
| `-l`/`--lower` | Minimum value for grid data | -1 |
| `-u`/`--upper` | Maximum value for grid data | 1 |
| `--no-labels` | Disables label generation | Enabled |

**Spiral-Specific Parameters**

- `-n`: Controls radial noise (default: 0.5)
- `-r`: Sets number of full rotations (default: 0.5)

**Output Details**

Generated JSON files are saved to `demos/demo-data/[TIMESTAMP].json`. Each file contains:
- `data`: Array of coordinate pairs (e.g., `[[0.34, -0.12], [0.88, 0.24]]`)
- `labels`: Numeric class identifiers (omitted if `--no-labels` is used)

**Grid Mode Notes**

- Works in any dimensionality (adjust with `-d`)
- Labels use `types^dimension` classification logic for grid generation

### 🎥 Visualizer

## 🔁 Export & Interoperate
| Format | Module | Notes |
|--------|--------|-------|
| **ONNX** | `del-onnx.*` | Compatible with PyTorch, TensorFlow, ORT |
| **JSON** | `del-export.*` | Human‑readable weights, includes visualization grid |
| **YAML** | `del-yaml.*` | Config‑driven experiments |
