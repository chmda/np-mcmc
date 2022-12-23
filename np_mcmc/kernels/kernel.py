from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, Optional, Tuple

from np_mcmc.types import JitMethod

import numpy as np

__all__ = ["MCMCKernel", "State"]

State = NamedTuple
_Kernel = Callable[[Tuple], Tuple]


class MCMCKernel(ABC):
    """Markov transition kernel that is usef for :class:`~np_mcmc.mcmc.MCMC`."""

    @abstractmethod
    def init(
        self,
        num_warmup: int,
        init_params: Optional[Tuple],
    ) -> Tuple[State, JitMethod]:
        """Initialize the `MCMCKernel` and return an initial state
        and the jit method to use.

        :param num_warmup: Number of warmup steps. This can be useful when doing adaptation.
        :type num_warmup: int
        :param init_params: Initial parameters to begin sampling.
        :type init_params: Optional[Tuple]
        :return: The initial state representing the state of the kernel and the jit method.
        :rtype: Tuple[State, JitMethod]
        """
        raise NotImplementedError

    @abstractmethod
    def get_sampler(self) -> _Kernel:
        """Returns the jitted Markov transition kernel

        :return: Transition kernel of signature `(tuple) -> tuple`
        :rtype: _Kernel
        """
        raise NotImplementedError

    @property
    def sample_field_idx(self) -> int:
        """Returns the index of the sample field in `State`.

        :return: Index of the sample field
        :rtype: int
        """
        raise NotImplementedError
