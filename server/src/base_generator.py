"""Base Module Generator.

This module is responsible for GeneratorAbstract and ModuleGenerator.
/opt/ml/ocrAPI/src/base_generator.py
- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from abc import ABC, abstractmethod
from typing import List, Union, Optional
from torch import nn as nn

def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ModuleGenerator:
    """Module generator class."""

    def __init__(self, module_name: str, in_channel: int):
        """Generate module based on the {module_name}

        Args:
            module_name: {module_name}Generator class must have been implemented.
        """
        self.module_name = module_name
        self.in_channel = in_channel

    def __call__(self, *args, **kwargs):
        # replace getattr
        return getattr(
            __import__("src.modules", fromlist=[""]),
            f"{self.module_name}Generator",
        )(self.in_channel, *args, **kwargs)
