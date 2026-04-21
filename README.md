# travel_py

Travel地面分割アルゴリズムの **Pythonリファレンス実装** です。

このリポジトリは、以下を目的とした **実験・デバッグ・可視化向け** 実装を提供します。

- アルゴリズム理解
- パラメータ感度分析
- 失敗ケース調査
- C++ / Rust移植前の高速プロトタイピング

⚠️ 本実装は **本番運用向けではありません**。性能最適化・リアルタイム性は対象外です。

---

## デザイン / デモ（GitHub Pages）

ブラウザで確認できるインタラクティブデモを公開しています。

- **Demo URL:** https://yoshiri.github.io/TRAVEL_ground_filter_python/interactive.html

> README上で示すデモURLは上記を正とします。

---

## 背景

既存のTravel実装はC++中心で実行性能に最適化されていることが多く、次の作業が難しい場合があります。

- ステップ単位のデバッグ
- 中間状態の可視化
- アルゴリズム試行錯誤

このプロジェクトでは、速度よりも **可読性・観測性・改造容易性** を優先しています。

---

## アルゴリズムパイプライン

実装はTravelの標準的な流れを段階的に保っています。

```text
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

各ステージをモジュール分割し、責務を明確化しています。

---

## プロジェクト構成

```text
travel_py/
├── src/travel_py/
│   ├── main.py           # パイプライン実行エントリ
│   ├── config.py         # パラメータ・閾値
│   ├── grid.py           # PointCloud → Gridマッピング
│   ├── cell_features.py  # セル特徴量計算
│   ├── adjacency.py      # 隣接関係構築
│   ├── traversal.py      # 地面伝播ロジック（中核）
│   ├── labeling.py       # セル→点ラベル反映
│   ├── debug_viz.py      # 可視化・デバッグ補助
│   └── types.py          # Enum / dataclass
├── pyproject.toml
└── README.md
```

---

## セットアップ（uv）

```bash
uv python install 3.11
uv python pin 3.11
uv venv
source .venv/bin/activate
uv sync
```

開発用Editableインストール:

```bash
uv pip install -e .
```

---

## 実行方法

```bash
python -m travel_py.main --points /path/to/points.npy
```

設定値は `config.py` と `configs/default.yaml` で管理しています。

---

## 設計方針

- Travelの段階的構造を維持する
- アルゴリズム本体と可視化を分離する
- 純関数と単純なデータ構造を優先する
- 1ファイル1責務を徹底する
- 読みやすく、壊しやすく、作り直しやすくする

---

## 非目標（意図的に未対応）

- ROS統合
- リアルタイム性能
- 並列化
- 抽象基底クラス中心の設計
- プラグインアーキテクチャ

アルゴリズムが安定してから必要に応じて追加します。

---

## 想定ワークフロー

1. Pythonでロジックを実装・調整
2. 中間状態（グリッド、伝播、除外理由）を可視化
3. ロジックとパラメータを安定化
4. 必要に応じてC++ / Rustへ移植

---

## テスト

```bash
# venv
uv sync
source .venv/bin/activate
uv pip install -e .

# test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

```bash
# ランダム点群で試験実行
python tools/make_sample.py
uv run python -m travel_py.main --points sample.npy --viz
```

---

## GitHub Pages向けデータ生成

`src/travel_py/main.py` はPython実行環境が必要なため、GitHub Pages（静的ホスティング）ではそのまま動作しません。  
このリポジトリでは、既存パイプラインの結果を静的JSONとして出力し、ブラウザ側で描画する方式を採用しています。

```bash
# 1) GitHub Pages用データを生成
uv run python tools/export_github_pages.py --points data/sample.npy

# 2) ローカル確認（例: Python標準HTTPサーバ）
python -m http.server 8000
# -> http://localhost:8000/docs/
```

生成物:
- `docs/data/demo_payload.json`: Ground / Non-ground推論結果付き点群
- `docs/index.html`: Plotlyベース静的3Dビューア

### GitHub Actionsで自動デプロイする最小手順

1. CIで `tools/export_github_pages.py` を実行し `docs/data/demo_payload.json` を更新
2. `docs/` をPages公開対象ブランチへデプロイ

`tools/rerun_debug.py` と同じ `travel_py.pipeline.run_pipeline` を使うため、
ローカル可視化とGitHub Pages可視化で同一推論結果を共有できます。

### 公開後の確認先

GitHub Pages有効化後、通常は以下で確認できます。

- `https://<user-or-org>.github.io/<repo-name>/`
- `https://<user-or-org>.github.io/<repo-name>/index.html`

本リポジトリの公開デモURL:

- **https://yoshiri.github.io/TRAVEL_ground_filter_python/interactive.html**

404の場合は **Settings > Pages** の公開ソース設定（`main` / `/docs` など）を確認してください。

---

## ライセンス

MIT License（または任意ライセンスを指定）。
