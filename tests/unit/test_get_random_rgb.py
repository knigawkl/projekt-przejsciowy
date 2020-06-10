from utils.colors import get_random_rgb


def test_get_random_rgb():
    colors = [get_random_rgb() for x in range(0, 100)]
    assert len(colors) == 100
