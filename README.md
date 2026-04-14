

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
python -m travel_py.main --points /path/to/points.npy
```

Configuration parameters are defined in `config.py` and `configs/default.yaml`.

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


## Testing

```bash
# venv
uv sync
source .venv/bin/activate
uv pip install -e .

# test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

```bash
# run test with random point
python tools/make_sample.py
uv run python -m travel_py.main --points sample.npy --viz
```

## GitHub Pages でのデモ公開

`src/travel_py/main.py` は Python 実行環境が必要なため、そのまま GitHub Pages（静的ホスティング）で直接は動きません。  
代わりに、このリポジトリでは **既存のパイプラインを再利用して静的JSONを出力し、ブラウザで描画する方式** を追加しました。

```bash
# 1) パイプラインを実行して GitHub Pages 用データを出力
uv run python tools/export_github_pages.py --points data/sample.npy

# 2) ローカル確認（例: Python標準HTTPサーバ）
python -m http.server 8000
# -> http://localhost:8000/docs/
```

生成物:
- `docs/data/demo_payload.json`: Ground / Non-ground の推論結果付き点群
- `docs/index.html`: Plotly ベースの静的3Dビューア

### GitHub Actions で自動デプロイする場合の最小手順

1. `tools/export_github_pages.py` を CI で実行して `docs/data/demo_payload.json` を更新
2. `docs/` を Pages 公開対象ブランチにデプロイ

この方式なら、`tools/rerun_debug.py` と同じ `travel_py.pipeline.run_pipeline` を使うため、
ローカルのRerun可視化と GitHub Pages 向け可視化で同じ推論結果を共有できます。

### merge 後にどこで見るか

GitHub Pages を有効化済みなら、merge 後は次の URL で見られます。

- ユーザー/組織ページ: `https://<user-or-org>.github.io/`
- プロジェクトページ: `https://<user-or-org>.github.io/<repo-name>/`

このリポジトリ構成では `docs/index.html` を公開しているため、通常は以下のどちらかです。

- `https://<user-or-org>.github.io/<repo-name>/`
- `https://<user-or-org>.github.io/<repo-name>/index.html`

もし 404 になる場合は、リポジトリの **Settings > Pages** で公開ソースを確認してください。

- Branch deploy の場合: `Branch = main` / `Folder = /docs`
- Actions deploy の場合: workflow が `docs/` を成果物として deploy していること

公開完了まで 1〜数分かかることがあります。反映後、`docs/data/demo_payload.json` を読み込んで 3D 表示されます。
