from collections.abc import Mapping
from typing import Any, Callable


def map_nested_dict(d: Any, func: Callable):

    if isinstance(d, Mapping):
        return {k: map_nested_dict(v, func) for k, v in d.items()}
    
    elif isinstance(d, (list, tuple)):
        return [map_nested_dict(v, func) for v in d]
    
    else:
        return func(d)