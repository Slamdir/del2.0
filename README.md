# DEL – Deep‑Learning Library for Ada  
*Team 19 Senior Design Project*

> **Mission** – deliver first‑class, fully type‑safe deep‑learning primitives to the Ada ecosystem.  

---

## 📑 Table of Contents
1. [Why DEL?](#-why-del)
2. [Prerequisites](#-prerequisites)
3. [Features](#-features)
4. [Quick Start](#-quick-start)
5. [Repository Layout](#-repository-layout)
6. [Framework Architecture](#-framework-architecture)
7. [Build & Install](#-build--install)
8. [Demos](#-demos)
9. [Usage](#-usage)
10. [Data Management](#-data-management)
11. [Model Management](#-model-management)
12. [Export & Interoperate](#-export--interoperate)
---

## 🚀 Why DEL?
Ada powers mission‑critical software (avionics, rail, defence, medical) yet lacks a native deep‑learning stack.  
Python‑based frameworks are powerful but introduce runtime GC, certification hurdles, and large footprints.

---

## 🔧 Prerequisites
| Component | Purpose | Source |
|-----------|---------|--------|
| **GNAT Ada Compiler** | Compiles Ada code | [AdaCore](https://www.adacore.com/download) |
| **Alire Package Manager** | Dependency & build manager | [Alire](https://alire.ada.dev/) |
| **Python 3.9+** *(optional)* | Runs demo/visual‑tools | [python.org](https://python.org) |
| **VS Code + Ada Extension** *(optional)* | IDE support | [VS Code](https://code.visualstudio.com/) |

**System requirements**  – Windows / Linux / macOS • ≥ 4 GB RAM (8 GB recommended) • ≥ 500 MB disk space.

> Install GNAT first, then run `alr get alire && alr setup` and verify with `alr --version`.

---

## ✨ Features
| Area | Module(s) | Highlights |
|------|-----------|-----------|
| Tensor core | `del.ads/adb` | N‑D tensors, broadcasting, slicing |
| Operators | `del-operators.*` | Vectorised matmul, element‑wise ops, reductions |
| Layers | via `del-model.*` | Dense, ReLU, Sigmoid, Softmax *(Conv & Pooling incoming)* |
| Loss functions | `del-loss.*` | MSE, Cross‑Entropy |
| Optimizers | `del-optimizers.*` | SGD, Momentum, Nesterov *(Adam & RMSProp planned)* |
| Utilities | `del-utilities.*` | RNG initialisers, progress bars, CLI helpers |
| Model container | `del-model.*` | Sequential & functional API, checkpointing |
| Data I/O | `del-data.*`, `del-json.*`, `del-yaml.*` | JSON ↔ Tensor, YAML config loader, synthetic data generator |
| Exporters | `del-export.*`, `del-onnx.*` | Flat‑file & **ONNX 1.15** emitters |

---

## 🚀 Quick Start
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
| **del-onnx** | ONNX v1.15 exporter |
| **del-export** | Flat JSON weight exporter |
| **del-utilities** | Misc helpers (initialisers, logging, CLI) |

---

## 🛠 Build & Install
### With **Alire** *(recommended)*
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
with DEL, DEL.Model, DEL.Optimizers, DEL.Loss;
procedure Hello_Del is
   use DEL;
   X : Tensor := Tensor'(Shape => (4,1), Data => (1,2,3,4));
   Y : Tensor := Tensor'(Shape => (4,1), Data => (0,1,0,1));

   Net : constant Model.Sequential :=
     Model.Sequential'
       (Model.Dense (1, 4, Init => Utilities.He_Normal),
        Model.ReLU,
        Model.Dense (4, 1),
        Model.Sigmoid);

   Opt  : Optimizers.SGD (0.01);
   Loss : constant Losses.Loss_Function := Losses.Binary_Cross_Entropy;

begin
   for Epoch in 1 .. 500 loop
      declare
         P : constant Tensor := Net (X);
         L : constant Float  := Loss (P, Y);
      begin
         Optimizers.Backprop (Opt, Net, L);
      end;
   end loop;
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
Model.Load_Data_From_File ("data.yaml",
                           Data_Shape   => (1,2),
                           Target_Shape => (1,1));
```
Use `Model.Export_To_JSON("output.json")` after training.

---

## 📦 Model Management
* **Checkpoint** – `Model.Save("run.chk")`, `Model.Load(...)`.
* **Config YAML** – define model/optimizer via `.yaml`, load with `del-yaml`.
* **Utilities** – progress bar, seed control, metric tracker in `del-utilities.*`.

---

## 🔁 Export & Interoperate
| Format | Module | Notes |
|--------|--------|-------|
| **ONNX 1.15** | `del-onnx.*` | Use in PyTorch, TensorFlow, ORT |
| JSON  | `del-export.*` | Human‑readable weights, schema in `docs/schema/` |
| YAML  | `del-yaml.*` | Config‑driven experiments |




