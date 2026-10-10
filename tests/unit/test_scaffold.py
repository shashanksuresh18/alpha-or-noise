import importlib

import pytest

SUBPACKAGES = ["engine", "detector", "alpha", "diffusion", "lab"]


@pytest.mark.unit
@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"alpha_or_noise.{name}")
