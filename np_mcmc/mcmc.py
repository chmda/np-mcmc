import warnings
import functools
from np_mcmc.kernels.kernel import MCMCKernel, State
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from numba import get_num_threads, prange, jit, njit

from np_mcmc.types import ChainMethod, JitMethod
from np_mcmc.utils import identity


def _single_chain_mcmc(
    init_state: Tuple, num_samples: int, sampler: Callable
) -> List[Tuple]:
    states: List[Tuple] = [()] * (num_samples + 1)
    states[0] = init_state
    for i in range(num_samples):
        states[i + 1] = sampler(states[i])
    return states


def _parallel_chains_mcmc(
    init_states: List[Tuple], num_samples: int, sampler: Callable, num_chains: int
) -> List[List[Tuple]]:
    states: List[List[Tuple]] = [None] * num_chains  # type: ignore
    q, r = divmod(num_samples, num_chains)
    n_samples_per_core: List[int] = [q] * num_chains
    n_samples_per_core[-1] += r
    for i in prange(num_chains):
        states[i] = _single_chain_mcmc(init_states[i], n_samples_per_core[i], sampler)
    return states


class MCMC:
    def __init__(
        self,
        kernel: MCMCKernel,
        num_warmup: int,
        num_chains: int = 1,
        chain_method: ChainMethod = "sequential",
    ) -> None:
        self._kernel: MCMCKernel = kernel
        self._num_warmup: int = num_warmup
        self._num_chains: int = num_chains
        if chain_method not in ["parallel", "sequential"]:
            raise ValueError(f"Chain method '{chain_method}' is not supported")
        if chain_method == "parallel" and get_num_threads() < self._num_chains:
            chain_method = "sequential"
            warnings.warn(
                f"There are not enough threads to run {self._num_chains} parallel chains."
                " Chains will be drawn sequentially."
                " You can get the number of threads using `numba.get_num_threads()`."
            )
        self._chain_method: ChainMethod = chain_method

        self._states: Optional[List[List[State]]] = None
        self._states_flat: Optional[List[State]] = None
        # returned by last run
        self._last_states: Optional[List[State]] = None
        # returned by last warmup
        self._warmup_states: Optional[List[State]] = None
        self._jit_method: JitMethod = "none"
        self._cached_fns: Dict[str, Dict[JitMethod, Callable]] = dict(
            single_chain_mcmc=dict(), parallel_chains_mcmc=dict()
        )

    @property
    def num_warmup(self) -> int:
        return self._num_warmup

    @property
    def num_chains(self) -> int:
        return self._num_chains

    @property
    def last_states(self) -> Optional[List[State]]:
        return self._last_states

    def run(
        self, num_samples: int, init_params: Optional[Union[List[Tuple], Tuple]] = None
    ) -> None:
        # check 'init_params' for parallel
        if init_params is not None and self._num_chains > 1:
            if not isinstance(init_params[0], (list, tuple)):
                raise ValueError(
                    "'init_params' must be a tuple or a list of tuples since 'num_chains > 1'"
                )
            n_tuples: Set[int] = set(len(item) for item in init_params)
            if len(n_tuples) != 1:
                raise ValueError(
                    "There must be the same number of elements in each tuple of 'init_params'"
                )
            if n_tuples.pop() != self._num_chains:
                raise ValueError(
                    "You must provide the same number of tuples as 'num_chains'"
                )

        init_states: Optional[List[State]] = self._last_states
        if self._warmup_states is not None:
            init_states = self._warmup_states
        states, last_states = self._run_mcmc(num_samples, init_states, init_params)
        states_flat: List[State] = sum(states, start=[])

        self._last_states = last_states
        self._states = states
        self._states_flat = states_flat

    def _run_mcmc(
        self,
        num_samples: int,
        init_states: Optional[List[State]],
        init_params: Optional[Union[List[Tuple], Tuple]],
    ) -> Tuple[List[List[State]], List[State]]:
        # init the kernel and get the appropriate jit method to use
        if init_states is None:
            init_states = []
            if init_params is not None and isinstance(init_params[0], (list, tuple)):
                for i in range(len(init_params)):
                    state, self._jit_method = self._kernel.init(
                        num_warmup=self._num_warmup,
                        init_params=init_params[i],
                    )
                    init_states.append(state)
            else:
                state, self._jit_method = self._kernel.init(
                    num_warmup=self._num_warmup,
                    init_params=init_params,  # type: ignore
                )
                init_states.append(state)

        # get jitter
        jitter: Callable = identity
        if self._jit_method == "jit":
            jitter = jit
        elif self._jit_method == "native":
            jitter = njit

        # get sampler
        sampler = self._kernel.get_sampler()

        # get chain functions
        if self._jit_method not in self._cached_fns["single_chain_mcmc"]:
            self._cached_fns["single_chain_mcmc"][self._jit_method] = jitter(
                functools.partial(_single_chain_mcmc, sampler=sampler)
            )
        jit_single_chain_mcmc = self._cached_fns["single_chain_mcmc"][self._jit_method]

        parallel_chain: bool = self._chain_method == "parallel"
        if parallel_chain:
            if self._jit_method not in self._cached_fns["parallel_chain_mcmc"]:
                self._cached_fns["parallel_chain_mcmc"][self._jit_method] = jitter(
                    functools.partial(
                        _parallel_chains_mcmc,
                        sampler=sampler,
                        num_chains=self._num_chains,
                    )
                )
            jit_parallel_chain_mcmc = self._cached_fns["parallel_chain_mcmc"][
                self._jit_method
            ]

        # run chains
        if parallel_chain:
            states = jit_parallel_chain_mcmc(init_states, num_samples)
        else:
            states_flat = jit_single_chain_mcmc(init_states[0], num_samples)
            states = [states_flat]
        return states, [states[i][-1] for i in range(len(states))]

    def warmup(self, init_params: Optional[Tuple] = None) -> None:
        self._warmup_states = None
        self.run(self._num_warmup * self._num_chains, init_params=init_params)
        self._warmup_states = self._last_states
