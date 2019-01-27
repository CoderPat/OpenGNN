import inspect
import json
from typing import Any, Callable
import gzip
import codecs


def read_file(filename):
    # TODO: better checks for filetype
    if filename.endswith('.gz'):
        return gzip.open(filename, mode='rt', encoding='utf-8')
    else:
        return open(filename, mode='rt', encoding='utf-8')


def read_jsonl_gz_file(filename):
    with read_file(filename=filename) as f:
        return [json.loads(line) for line in f]


def count_lines(filename: str) -> int:
    i = 0
    with read_file(filename) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


def find_max(filename: str, key: Callable[[Any], float]) -> float:
    max_atr = float("-inf")
    with read_file(filename) as f:
        for sample_txt in f.readlines():
            sample = json.loads(sample_txt)
            max_atr = max(max_atr, key(sample))
    return max_atr


def find_first(filename: str, key: Callable[[Any], float]) -> float:
    """ Extracts info about a dataset from the first sample """
    with read_file(filename) as f:
        first_sample = json.loads(f.readline())
    return key(first_sample)


def find(vector, item):
    for i, element in enumerate(vector):
        if item == element:
            return i
    return -1


def classes_in_module(module):
    """Returns a generator over the classes defined in :obj:`module`."""
    return (symbol for symbol in dir(module)
            if inspect.isclass(getattr(module, symbol)))
