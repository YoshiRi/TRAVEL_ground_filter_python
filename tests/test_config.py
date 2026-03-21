"""Tests for config.py — GlobalConfig defaults and YAML loading."""

import tempfile
from pathlib import Path

import pytest

from travel_py.config import (
    AcceptConfig,
    DebugConfig,
    GlobalConfig,
    GridConfig,
    SeedConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaultValues:
    def test_grid_config_defaults(self):
        cfg = GridConfig()
        assert cfg.resolution == 5
        assert cfg.origin_xy == (-50.0, -50.0)
        assert cfg.size_xy == (200, 200)

    def test_seed_config_defaults(self):
        cfg = SeedConfig()
        assert cfg.min_points == 1
        assert cfg.top_k == 5

    def test_accept_config_defaults(self):
        cfg = AcceptConfig()
        assert cfg.max_height_diff == 0.5
        assert cfg.th_normal == 0.9
        assert cfg.max_slope is None

    def test_debug_config_defaults(self):
        cfg = DebugConfig()
        assert cfg.enable_viz is False

    def test_global_config_creates_sub_configs(self):
        cfg = GlobalConfig()
        assert isinstance(cfg.grid, GridConfig)
        assert isinstance(cfg.seed, SeedConfig)
        assert isinstance(cfg.accept, AcceptConfig)
        assert isinstance(cfg.debug, DebugConfig)


# ---------------------------------------------------------------------------
# load_config with no path → returns defaults
# ---------------------------------------------------------------------------

class TestLoadConfigNoPath:
    def test_returns_default_config(self):
        cfg = load_config(None)
        assert isinstance(cfg, GlobalConfig)
        assert cfg.grid.resolution == GridConfig().resolution

    def test_nonexistent_default_config_path_uses_defaults(self, tmp_path):
        # When the path doesn't exist, load_config will raise FileNotFoundError
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(missing)


# ---------------------------------------------------------------------------
# load_config with YAML
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


class TestLoadConfigFromYaml:
    def test_empty_yaml_returns_defaults(self, tmp_path):
        p = _write_yaml(tmp_path, "")
        cfg = load_config(p)
        assert cfg.grid.resolution == GridConfig().resolution

    def test_override_grid_resolution(self, tmp_path):
        p = _write_yaml(tmp_path, "grid:\n  resolution: 10.0\n")
        cfg = load_config(p)
        assert cfg.grid.resolution == 10.0
        # Other grid fields stay default
        assert cfg.grid.origin_xy == (-50.0, -50.0)

    def test_override_grid_origin(self, tmp_path):
        p = _write_yaml(tmp_path, "grid:\n  origin_xy: [0.0, 0.0]\n")
        cfg = load_config(p)
        assert list(cfg.grid.origin_xy) == [0.0, 0.0]

    def test_override_seed_top_k(self, tmp_path):
        p = _write_yaml(tmp_path, "seed:\n  top_k: 10\n")
        cfg = load_config(p)
        assert cfg.seed.top_k == 10

    def test_override_accept_max_height_diff(self, tmp_path):
        p = _write_yaml(tmp_path, "accept:\n  max_height_diff: 0.3\n")
        cfg = load_config(p)
        assert cfg.accept.max_height_diff == pytest.approx(0.3)

    def test_override_accept_th_normal(self, tmp_path):
        p = _write_yaml(tmp_path, "accept:\n  th_normal: 0.95\n")
        cfg = load_config(p)
        assert cfg.accept.th_normal == pytest.approx(0.95)

    def test_override_debug_enable_viz(self, tmp_path):
        p = _write_yaml(tmp_path, "debug:\n  enable_viz: true\n")
        cfg = load_config(p)
        assert cfg.debug.enable_viz is True

    def test_partial_override_does_not_affect_others(self, tmp_path):
        p = _write_yaml(tmp_path, "grid:\n  resolution: 3.0\n")
        cfg = load_config(p)
        # seed and accept should still be defaults
        assert cfg.seed.top_k == SeedConfig().top_k
        assert cfg.accept.max_height_diff == AcceptConfig().max_height_diff

    def test_multiple_sections(self, tmp_path):
        yaml_content = """\
grid:
  resolution: 4.0
seed:
  top_k: 7
accept:
  max_height_diff: 0.2
  th_normal: 0.85
debug:
  enable_viz: false
"""
        p = _write_yaml(tmp_path, yaml_content)
        cfg = load_config(p)
        assert cfg.grid.resolution == 4.0
        assert cfg.seed.top_k == 7
        assert cfg.accept.max_height_diff == pytest.approx(0.2)
        assert cfg.accept.th_normal == pytest.approx(0.85)
        assert cfg.debug.enable_viz is False
