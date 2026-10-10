"""FND-05 픽스처."""
import pytest

from fnd05_helpers import load


@pytest.fixture(scope="session")
def lc():
    return load()
