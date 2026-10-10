"""RUN-05 공용 픽스처."""
import pytest
from run05_helpers import Box


@pytest.fixture
def box(tmp_path):
    return Box(tmp_path)
