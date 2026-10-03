"""Smoke test: package imports and base config loads."""

from catalogiq.config import get_config, resolve_path
from catalogiq.utils import set_seed


def test_package_imports() -> None:
    import catalogiq

    assert catalogiq is not None


def test_config_loads() -> None:
    cfg = get_config()
    assert cfg["project"]["name"] == "catalogiq"
    assert isinstance(cfg["project"]["seed"], int)
    assert cfg["models"]["llm"]["model"] == "llama3.2:3b"
    assert resolve_path(cfg["paths"]["data_raw"]).parts[-2:] == ("data", "raw")


def test_set_seed_is_reproducible() -> None:
    import numpy as np

    set_seed(1)
    a = np.random.rand(3)
    set_seed(1)
    b = np.random.rand(3)
    assert (a == b).all()
