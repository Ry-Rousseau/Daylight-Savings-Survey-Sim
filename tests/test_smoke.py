"""Unit 0.1 smoke: the package installs editable and imports."""

import polis


def test_polis_importable():
    assert hasattr(polis, "__version__")
