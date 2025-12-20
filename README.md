

# travel_py

Python-based reference implementation of the **Travel ground segmentation algorithm**.

This repository provides an **experimental, debuggable, and visualization-friendly**
implementation intended for:

- Algorithm understanding
- Parameter sensitivity analysis
- Failure case investigation
- Rapid prototyping before C++ / Rust implementations

⚠️ This is **not production code**. Performance and real-time constraints are out of scope.

---

## Motivation

Existing Travel implementations are often written in C++ and optimized for runtime,
which makes:

- Step-by-step debugging
- Intermediate state visualization
- Algorithmic trial-and-error

relatively difficult.

This project prioritizes **clarity, inspectability, and modifiability** over speed.

---

## Algorithm Pipeline

The implementation strictly preserves the canonical Travel pipeline:

```

PointCloud Input
↓
Grid Builder
↓
Cell Feature Extraction
↓
Adjacency Construction
↓
Traversal (Ground Propagation)
↓
Point Labeling
↓
Debug / Visualization

```

Each stage is isolated in its own module to keep responsibilities explicit.

---

## Project Structure

```

travel_py/
├── src/travel_py/
│   ├── main.py          # Pipeline orchestration
│   ├── config.py        # Parameters and thresholds
│   ├── grid.py          # PointCloud → Grid mapping
│   ├── cell_features.py# Cell-level feature computation
│   ├── adjacency.py    # Neighbor relationships
│   ├── traversal.py    # Ground propagation logic (core)
│   ├── labeling.py     # Cell → point label propagation
│   ├── debug_viz.py    # Visualization & debugging utilities
│   └── types.py        # Enums and dataclasses
├── pyproject.toml
└── README.md

````

---

## Environment Setup (uv)

This project uses **uv** for fast and reproducible environment management.

```bash
uv python install 3.11
uv python pin 3.11
uv venv
source .venv/bin/activate
uv sync
````

Editable install for development:

```bash
uv pip install -e .
```

---

## Running

```bash
python -m travel_py.main
```

Configuration parameters are defined in `config.py`
(or externalized later if needed).

---

## Design Principles

* Preserve Travel's stage-wise structure
* Separate algorithm logic from visualization
* Favor pure functions and simple data structures
* One file = one responsibility
* Easy to read, easy to break, easy to rebuild

---

## Non-goals (by design)

* ROS integration
* Real-time performance
* Parallelization
* Abstract base class hierarchies
* Plugin architectures

These may be added **after** the algorithm stabilizes.

---

## Intended Workflow

1. Implement and modify logic in Python
2. Visualize intermediate states (grid, traversal, rejection reasons)
3. Identify stable logic and parameters
4. Port validated logic to C++ / Rust if needed

---

## License

MIT License (or specify your preferred license).


## GitHub 登録までの最短手順

```bash
git init
git add README.md .gitignore pyproject.toml src
git commit -m "Initial commit: Python Travel ground segmentation reference implementation"

git branch -M main
git remote add origin git@github.com:<your_name>/travel_py.git
git push -u origin main
```
