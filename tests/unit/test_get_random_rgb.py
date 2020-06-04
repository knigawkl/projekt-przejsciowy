from utils.colors import get_random_rgb


def test_get_random_rgb():
    dupa = [get_random_rgb() for x in range(0, 100)]
    print(dupa)
    print(get_random_rgb())

test_get_random_rgb()
