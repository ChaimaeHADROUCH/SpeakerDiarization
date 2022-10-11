# This code comes from pyannote.audio
# All credits go to the pyannote.audio author(s)
# we don't use pyannote.audio here so we took just some useful functions
import itertools
from string import ascii_uppercase
from typing import Iterable, Union, List, Set, Optional, Iterator

def merge_dict(defaults: dict, custom: dict = None):
    params = dict(defaults)
    if custom is not None:
        params.update(custom)
    return params

def pairwise(iterable: Iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
