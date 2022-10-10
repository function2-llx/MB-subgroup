import os
# python does not support intersection type at the time (https://github.com/python/typing/issues/213)
# with this way, at least suggestions in IDE will be available
from typing import Union as Intersection, Union

PathLike = Union[str, bytes, os.PathLike]

SEG_PROB_FILENAME = 'seg-prob.pt'

__all__ = ['Intersection', 'PathLike', 'SEG_PROB_FILENAME']
