import re
import math


NUMBERS = re.compile(r'(\d+)')


def get_2D_dist(x1, y1, x2, y2):
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def sort_num(value):
    parts = NUMBERS.split(value)
    parts[1::2] = list(map(int, parts[1::2]))
    return parts
