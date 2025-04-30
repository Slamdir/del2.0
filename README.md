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
9. [Usage](#-usage)
10. [Data Management](#-data-management)
11. [Model Management](#-model-management)
12. [Export & Interoperate](#-export--interoperate)
---

## 🚀 Why DEL?
Ada powers mission‑critical software (avionics, rail, defence, medical) yet lacks a native deep‑learning stack.  
Python‑based frameworks are powerful but introduce runtime GC, certification hurdles, and large footprints.

---

## 🔧 Prerequisites
| Component | Purpose | Source |
|-----------|---------|--------|
| **GNAT Ada Compiler** | Compiles Ada code | [AdaCore](https://www.adacore.com/download) |
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
git clone https://github.com/<org>/del.git
cd del
alr build              # or: gprbuild -Pdel.gpr
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
alr publish --installdir ~/.local
```
### With **GNAT**
```bash
gprbuild -Pdel.gpr -Xsuppress_gnat_style_warnings=true
```
Set `GNAT_PROJECT_PATH` if installing globally.

---

## 🎓 Demos
```bash
# Generate synthetic classification data
python3 demos/generator.py --samples 500 --classes 3
```

---

## 📝 Usage
Minimal example:
```ada
with Del; use Del;
with Del.Model;
with Del.Operators; use Del.Operators;
with Del.Optimizers;
with Del.Loss;

procedure Hello_Del is
   -- Create model instance
   My_Model : Del.Model.Model;
   
   -- Create and configure SGD optimizer
   SGD      : Del.Optimizers.SGD_T := 
              Del.Optimizers.Create_SGD_T(
                Learning_Rate => 0.01,
                Weight_Decay  => 0.0,
                Momentum      => 0.9);
   
   -- Create loss function
   CE_Loss  : Del.Loss.Cross_Entropy_T;
   
   -- Sample data
   X : Tensor_T := Zeros((4, 2));  -- Input data shape (4 samples, 2 features)
   Y : Tensor_T := Zeros((4, 3));  -- Target data shape (4 samples, 3 classes)
begin
   -- Add layers to model
   My_Model.Add_Layer(new Linear_T);   -- Linear layer
   My_Model.Add_Layer(new ReLU_T);     -- ReLU activation
   My_Model.Add_Layer(new Linear_T);   -- Output layer
   My_Model.Add_Layer(new SoftMax_T);  -- SoftMax for classification
   
   -- Configure model
   My_Model.Add_Loss(CE_Loss'Access);
   My_Model.Set_Optimizer(SGD'Access);
   
   -- Load data (typically from file)
   -- My_Model.Load_Data_From_File("data.json", Data_Shape => (1,2), Target_Shape => (1,3));
   
   -- Train model
   My_Model.Train_Model(
      Batch_Size => 4,
      Num_Epochs => 100);
   
   -- Run inference
   declare
      Prediction : Tensor_T := My_Model.Run_Layers(X);
   begin
      null; -- Process prediction
   end;
end Hello_Del;
```
Compile with `gnatmake hello_del.adb -Pdel.gpr`.

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

## 🔁 Export & Interoperate
| Format | Module | Notes |
|--------|--------|-------|
| **ONNX** | `del-onnx.*` | Compatible with PyTorch, TensorFlow, ORT |
| **JSON** | `del-export.*` | Human‑readable weights, includes visualization grid |
| **YAML** | `del-yaml.*` | Config‑driven experiments |