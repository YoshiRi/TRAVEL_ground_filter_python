# TRAVEL Ground Filter Python — 開発メモ

## 環境セットアップ

```bash
# uv が未インストールの場合
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 初回セットアップ
uv python install 3.11
uv python pin 3.11
uv venv
uv sync
uv pip install -e .

# 実行
uv run python tools/make_sample.py          # サンプルデータ生成
uv run python -m travel_py.main --points sample.npy           # パイプライン実行
uv run python -m travel_py.main --points sample.npy --viz     # 可視化あり

# テスト
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

---

## 直近の作業（2026-03-28 完了）

1. **`configs/default.yaml`** — グリッド設定を点群範囲に合わせて修正
   - origin_xy: `[-50, -50]` → `[-85, -85]`、size_xy: `[200, 200]` → `[170, 170]`
   - 修正前は点群の27%がグリッド外に落ちており、Unknown領域の主因だった

2. **完了確認**（コミット `9235a85` で実装済みだったことを確認）
   - `graph.py`: LOOSEな32近傍 → 厳密なTGS辺共有接続（内部3 + 外部1）に置き換え済み
   - `seed.py`: `top_k` パラメータ対応済み（現在top_k=5で動作確認）
   - `config.py` / `main.py`: `AcceptConfig.th_normal` / `top_k` のconfig統合済み

---

## 直近の作業（2026-03-20 完了）

1. **`main.py`** — パイプラインを完成させた
2. **`config.py`** — Python 3.11 対応
3. **`seed.py`** — グリッドセンター距離フィルタ削除

---

## 残りのTODO（優先度順）

### 高優先度

#### 1. テストの追加
以下のモジュールにテストがない：
- `src/travel_py/plane.py`（特に `is_traversable_lcc()` の境界値テスト）
- `src/travel_py/traversal.py`
- `src/travel_py/seed.py`
- `src/travel_py/graph.py`
- `src/travel_py/accept.py`

### 中優先度

#### 2. 空間分散シード選択（`seed.py`）
- **現状**: weight降順top_kのため、高weightなSubCellが空間的に集中していると全シードが一箇所に固まる
- **目標**: 角度セクタ分割など空間的に分散したシード選択
- **議論メモ**: ロボティクス用途では近傍Groundが最重要で遠方シードは計算対効果が低い。Optionとして追加する程度が適切

#### 3. SubCell への点インデックス保持（精度向上）
- **現状**: `SubCell.points` は座標コピーのみ（元インデックスなし）
- SubCell ごとに元インデックスを持てば、より細粒度のラベル伝播が可能
- **ファイル**: `src/travel_py/types.py`, `src/travel_py/grid.py`, `src/travel_py/labeling.py`

---

## アーキテクチャ概要

```
main.py          パイプライン統括
config.py        設定（dataclass + YAML）
types.py         Cell, SubCell, SubCellIndex, CellState
grid.py          点群 → グリッド変換、SubCell分割
cell_features.py Cell の高さ統計（min/max/mean/range）
plane.py         局所平面推定（LPR + PCA）、LCC判定
graph.py         SubCell 隣接グラフ
seed.py          シード選択
traversal.py     BFS（SubCellベース）
accept.py        旧セルベース受理基準（現在未使用）
labeling.py      Cell.state → 点ラベル伝播
debug_viz.py     Matplotlib 可視化
```

## パイプライン実行フロー

```
[1] Grid.from_points()             → Cell + SubCell 構築
[2] compute_all_cell_features()    → Cell の高さ統計
[3] LocalPlaneEstimator            → SubCell ごとに法線・重み・GROUND/NON_GROUND判定
[4] TraversabilityGraph()          → SubCell 隣接グラフ
[5] find_dominant_subcells()       → BFS 開始ノード選択
[6] run_subcell_traversal()        → LCC基準でBFS、到達SubCellをGROUNDに
[7] SubCell.label → Cell.state     → 伝播
[8] label_points_from_cells()      → 点ごとのラベル(1/0/-1)
```
