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
uv run python -m travel_py.main --points sample.npy               # 高速ベクトル化パイプライン実行 (デフォルト)
uv run python -m travel_py.main --points sample.npy --reference   # 参照実装 (オブジェクトベース) 実行
uv run python -m travel_py.main --points sample.npy --viz         # 可視化あり

# テスト
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

---

## 直近の作業（2026-11 公式C++準拠 Adaptive LCC 実装 `feat/adaptive-lcc`）

1. **距離適応型 LCC（Adaptive Local Convexity / Concavity）の導入**
   - 公式 C++ 実装 (`darrenjkt/TRAVEL` / `tgs.hpp` L816-834) に準拠した距離比例の適応閾値判定を実装
   - `similarity >= 1 - sin(d * th_lcc_normal_similarity)`（角度係数 デフォルト 0.03）
   - `max_dist = d * sin(th_lcc_planar_dist)`（角度係数 デフォルト 0.2）
   - `AcceptConfig.use_adaptive_lcc` で固定閾値モード（既存）と適応閾値モードの切り替えに対応
   - `src/travel_py/plane.py`、`fast.py`、`pipeline.py`、`config.py` に適用し、両モードでの高速パス・参照パスの完全等価性をテストで検証

---

## 直近の作業（2026-11 高速化対応 `perf/vectorize` PR #19 マージ完了）


1. **ベクトル化パイプラインの実装 (`src/travel_py/fast.py`, `pipeline.py`)**
   - 点群を `(sid, z)` でソートしたフラット配列（SoA: Structure of Arrays）により、点ごと・SubCellごとのPythonループを排除
   - LPRおよび共分散計算を `np.bincount` とバッチ `np.linalg.eigh` でベクトル化
   - BFSトラバーサルもフラットインデックスで高速実行
   - 処理時間を 121ms → 61ms（解像度5m）、557ms → 67ms（解像度1m）、1049ms → 83ms（解像度0.5m）へ高速化（2.4〜12.6倍高速化）

2. **既存オブジェクト構造との互換性・等価性の担保**
   - `grid.py`: `Grid.from_points` を内部ベクトル化しつつ、従来の `Cell` / `SubCell` オブジェクト生成を維持
   - `main.py`: デフォルトで高速化パスを実行し、`--reference` で従来のオブジェクトベースパイプラインに切り替え可能。可視化時（`--viz`）のみ `to_grid()` でデバッグ表示オブジェクトを復元
   - `tests/test_fast_equivalence.py`: 参照実装と高速化実装が合成点群・実点群データ（sample.npy）において点ラベル単位で1点も狂わず完全一致することを検証（全テストパス）

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

## アーキテクチャ概要

```
main.py          パイプライン統括 (--reference で従来実装、デフォルトは fast)
pipeline.py      run_fast / run_reference を提供する統合インターフェース
fast.py          SoA (FlatGrid) による完全ベクトル化パイプライン
config.py        設定（dataclass + YAML）
types.py         Cell, SubCell, SubCellIndex, CellState
grid.py          点群 → グリッド変換、SubCell分割
cell_features.py Cell の高さ統計（min/max/mean/range）
plane.py         局所平面推定（LPR + PCA）、LCC判定
graph.py         SubCell 隣接グラフ
seed.py          シード選択
traversal.py     BFS（SubCellベース）
labeling.py      Cell.state → 点ラベル伝播
debug_viz.py     Matplotlib 可視化
```
