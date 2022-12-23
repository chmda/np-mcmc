from typing import Any, Callable, Dict, Union, TypeVar, List
from np_mcmc.types import JitMethod
from numba.extending import is_jitted
from numba import jit, njit

T = TypeVar("T")
U = TypeVar("U")

"""
def check_random_state(seed: RngKey) -> np.random.Generator:
    if seed is None or isinstance(
        seed,
        (
            numbers.Integral,
            np.ndarray,
            list,
            np.random.SeedSequence,
            np.random.BitGenerator,
            np.random.Generator,
        ),
    ):
        return np.random.default_rng(seed)  # type: ignore
    elif isinstance(seed, np.random.RandomState):
        return np.random.default_rng(seed._bit_generator)
    raise ValueError(
        f"{seed!r} cannot be used to seed a numpy.random.Generator instance"
    )
"""


def get_jit_method(func: Callable) -> JitMethod:
    if not is_jitted(func):
        return "none"
    target_options: Dict = getattr(func, "targetoptions", dict())
    nopython: bool = target_options.get("nopython", False)
    if nopython or len(getattr(func, "nopython_signatures", [])) > 0:
        return "native"

    return "jit"


def identity(x: Any) -> Any:
    return x


def get_jitter(method: JitMethod) -> Callable:
    jitter: Callable = identity
    if method == "jit":
        jitter = jit
    elif method == "native":
        jitter = njit
    return jitter
