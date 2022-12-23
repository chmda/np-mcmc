from typing import Callable, NamedTuple, Optional, Tuple
from np_mcmc.kernels.kernel import MCMCKernel, State, _Kernel
from np_mcmc.types import JitMethod, PotentialFunc
import numpy as np

from np_mcmc.utils import get_jit_method, get_jitter

MALAState = NamedTuple("MALAState", [("u", np.ndarray)])


class MALA(MCMCKernel):
    r"""Defines the Markov kernel of Metropolis-adjusted Langevin algorithm.

    Let :math:`\pi : \mathbb R^d \to \mathbb R` the probability density function from which
    it is desired to draw an ensemble of i.i.d. samples.
    We consider the Langevin Itô diffusion

    .. math::

        \mathrm{d}X_t = \nabla \log \pi(X_t) \mathrm{d}t + \sqrt{2}\mathrm{d}B_t

    where :math:`B_t` denotes the standard d-dimensional Brownian motion.

    Approximate sample paths of the diffusion can be generated using Eueler-Maruyama method
    with a fixed time step :math:`\tau > 0` by

    .. math::

        X_{t+1} := X_t + \tau\nabla \log\pi(X_t) + \sqrt{2\tau}\xi_t

    where :math:`\xi_t \sim \mathcal N_d(0, I_d)`.
    We define the proposal :math:`\tilde X_{t+1} := X_{t+1}` as a proposal for a new state.

    The proposal is accepted or rejected according to the Metropolis-Hastings algorithm

    .. math::

        \alpha_{t+1} := \min\left(1, \frac{\pi(\tilde X_{t+1})q(X_t | \tilde X_{t+1})}{\pi(\tilde X_t)q(\tilde X_{t+1} | X_t)})

    where

    .. math::

        q(x' | x) \propto \exp(-\frac{1}{4\tau}\Vert x'-x-\tau\nabla\log\pi(x)\Vert_2^2)

    is the transition probability density from :math:`x` to :math:`x'`.
    """
    _jit_method: JitMethod
    sample_field_idx: int = 0

    def __init__(
        self,
        potential_fn: PotentialFunc,
        grad_potential_fn: PotentialFunc,
        step_size: float = 0.1,
    ) -> None:
        """Creates the Markov kernel

        :param potential_fn: Potential function defined as :math:`\log\pi`.
        :type potential_fn: PotentialFunc
        :param grad_potential_fn: Gradient of the potential function
        :type grad_potential_fn: PotentialFunc
        :param step_size: Step size in Euler-Maruyama method, defaults to 0.1
        :type step_size: float, optional
        """
        super().__init__()
        self._potential_fn: PotentialFunc = potential_fn
        self._grad_potential_fn: PotentialFunc = grad_potential_fn
        self._step_size: float = step_size
        self._kernel: _Kernel = None

    def init(
        self, num_warmup: int, init_params: Optional[Tuple]
    ) -> Tuple[State, JitMethod]:
        if init_params is None:
            raise ValueError("You must provide 'init_params'")
        (u_init,) = init_params
        self._jit_method = get_jit_method(self._potential_fn)
        if get_jit_method(self._grad_potential_fn) != self._jit_method:
            raise ValueError(
                "'potential_fn' and 'grad_potential_fn' have not the same jitter"
            )
        return MALAState(u_init), self._jit_method

    def get_sampler(self) -> _Kernel:
        jitter: Callable = get_jitter(self._jit_method)
        if self._kernel is None:
            if self._jit_method == "none":

                def kernel(
                    state: Tuple,
                    step_size: float = self._step_size,
                    potential_fn: Callable = self._potential_fn,
                    grad_potential_fn: Callable = self._grad_potential_fn,
                ) -> Tuple:
                    (u,) = state
                    u_proposal: np.ndarray = np.sqrt(2 * step_size) * np.random.normal(
                        loc=u + step_size * grad_potential_fn(u)
                    )
                    frac_transition_prob: np.ndarray = -(
                        np.linalg.norm(
                            u - u_proposal - step_size * grad_potential_fn(u_proposal)
                        )
                        ** 2
                        - np.linalg.norm(
                            u_proposal - u - step_size * grad_potential_fn(u)
                        )
                        ** 2
                    ) / (4 * step_size)
                    accept_prob: float = np.exp(
                        potential_fn(u_proposal)
                        - potential_fn(u)
                        + frac_transition_prob
                    )
                    if np.random.rand() < accept_prob:
                        return (u_proposal,)
                    return (u,)

            else:

                def kernel(
                    state: Tuple,
                    step_size: float = self._step_size,
                    potential_fn: Callable = self._potential_fn,
                    grad_potential_fn: Callable = self._grad_potential_fn,
                ) -> Tuple:
                    (u,) = state
                    u_proposal: np.ndarray = np.zeros_like(u)
                    mu: np.ndarray = u + step_size * grad_potential_fn(u)
                    for i in range(u.shape[0]):
                        u_proposal[i] = np.sqrt(2 * step_size) * np.random.normal(
                            loc=mu[i]
                        )
                    frac_transition_prob: np.ndarray = -(
                        np.linalg.norm(
                            u - u_proposal - step_size * grad_potential_fn(u_proposal)
                        )
                        ** 2
                        - np.linalg.norm(
                            u_proposal - u - step_size * grad_potential_fn(u)
                        )
                        ** 2
                    ) / (4 * step_size)
                    accept_prob: float = np.exp(
                        potential_fn(u_proposal)
                        - potential_fn(u)
                        + frac_transition_prob
                    )
                    if np.random.rand() < accept_prob:
                        return (u_proposal,)
                    return (u,)

            self._kernel = jitter(kernel)
        return self._kernel
