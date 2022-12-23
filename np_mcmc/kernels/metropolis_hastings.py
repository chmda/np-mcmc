from typing import Callable, NamedTuple, Optional, Tuple
from np_mcmc.kernels.kernel import MCMCKernel, State, _Kernel
from np_mcmc.types import JitMethod, PotentialFunc
import numpy as np

from np_mcmc.utils import get_jit_method, get_jitter

MHState = NamedTuple("MHState", [("u", np.ndarray)])


class MetropolisHastings(MCMCKernel):
    _jit_method: JitMethod
    sample_field_idx: int = 0

    def __init__(self, potential_fn: PotentialFunc, step_size: float = 0.1) -> None:
        super().__init__()
        self._potential_fn: PotentialFunc = potential_fn
        self._step_size: float = step_size
        self._kernel: _Kernel = None

    def init(
        self, num_warmup: int, init_params: Optional[Tuple]
    ) -> Tuple[State, JitMethod]:
        if init_params is None:
            raise ValueError("You must provide 'init_params'")
        (u_init,) = init_params
        self._jit_method = get_jit_method(self._potential_fn)
        return MHState(u_init), self._jit_method

    def get_sampler(self) -> _Kernel:
        jitter: Callable = get_jitter(self._jit_method)
        if self._kernel is None:
            if self._jit_method == "none":

                def kernel(
                    state: Tuple,
                    step_size: float = self._step_size,
                    potential_fn: Callable = self._potential_fn,
                ) -> Tuple:
                    (u,) = state
                    u_proposal: np.ndarray = np.random.normal(loc=u, scale=step_size)
                    accept_prob: float = np.exp(
                        potential_fn(u_proposal) - potential_fn(u)
                    )
                    if np.random.rand() < accept_prob:
                        return (u_proposal,)
                    return (u,)

            else:

                def kernel(
                    state: Tuple,
                    step_size: float = self._step_size,
                    potential_fn: Callable = self._potential_fn,
                ) -> Tuple:
                    (u,) = state
                    u_proposal: np.ndarray = np.zeros_like(u)
                    for i in range(u.shape[0]):
                        u_proposal[i] = np.random.normal(loc=u[i], scale=step_size)
                    accept_prob: float = np.exp(
                        potential_fn(u_proposal) - potential_fn(u)
                    )
                    if np.random.rand() < accept_prob:
                        return (u_proposal,)
                    return (u,)

            self._kernel = jitter(kernel)
        return self._kernel
