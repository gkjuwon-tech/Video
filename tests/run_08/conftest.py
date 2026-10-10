import pytest
from run08_helpers import load_module


@pytest.fixture(scope="session")
def t4():
    return load_module()
