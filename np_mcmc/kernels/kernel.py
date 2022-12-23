from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, Optional, Tuple

from np_mcmc.types import JitMethod

import numpy as np

__all__ = ["MCMCKernel", "State"]

State = NamedTuple
_Kernel = Callable[[Tuple], Tuple]


class MCMCKernel(ABC):
    @abstractmethod
    def init(
        self,
        num_warmup: int,
        init_params: Optional[Tuple],
    ) -> Tuple[State, JitMethod]:
        raise NotImplementedError

    @abstractmethod
    def get_sampler(self) -> _Kernel:
        raise NotImplementedError

    @property
    def sample_field_idx(self) -> int:
        raise NotImplementedError
