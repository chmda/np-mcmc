from typing import Callable, NamedTuple, Optional, Tuple
from np_mcmc.kernels.kernel import MCMCKernel, State, _Kernel
from np_mcmc.types import JitMethod, PotentialFunc
import numpy as np

from np_mcmc.utils import get_jit_method, get_jitter

MHState = NamedTuple("MHState", [("u", np.ndarray)])


class MetropolisHastings(MCMCKernel):
    r"""Defines the Markov kernel of Metropolis-Hastings algorithm.

    Let :math:`\pi : \mathbb R^d \to \mathbb R` the probability density function from which
    it is desired to draw an ensemble of i.i.d. samples.

    Given a fixed time step :math:`\tau > 0`, a new proposal is given by

    .. math::

        \tilde X_{t+1} := X_t + \sqrt{\tau}\xi_t

    where :math:`\xi_t \sim \mathcal N_d(0, I_d)`.

    The proposal is accepted or rejected according to the Metropolis-Hastings algorithm

    .. math::

        \alpha_{t+1} := \min\left(1, \frac{\pi(\tilde X_{t+1})}{\pi(\tilde X_t)})
    """
    _jit_method: JitMethod
    sample_field_idx: int = 0

    def __init__(self, potential_fn: PotentialFunc, step_size: float = 0.1) -> None:
        """Creates the Markov kernel

        :param potential_fn: Potential function defined as :math:`\log\pi`.
        :type potential_fn: PotentialFunc
        :param step_size: Standard deviation of the proposal, defaults to 0.1
        :type step_size: float, optional
        """
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
                    u_proposal: np.ndarray = step_size * np.random.normal(loc=u)
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
                        u_proposal[i] = step_size * np.random.normal(loc=u[i])
                    accept_prob: float = np.exp(
                        potential_fn(u_proposal) - potential_fn(u)
                    )
                    if np.random.rand() < accept_prob:
                        return (u_proposal,)
                    return (u,)

            self._kernel = jitter(kernel)
        return self._kernel
