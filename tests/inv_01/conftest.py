"""INV-01 공용 픽스처. 방 30개는 세션당 한 번만 만든다(CPU 시간 예산)."""
import pytest


@pytest.fixture(scope="session")
def rooms():
    from buc.invent.grammar.room import generate_room
    return [generate_room(s) for s in range(30)]
