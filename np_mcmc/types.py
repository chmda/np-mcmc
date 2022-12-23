from typing import Callable, Literal
import numpy as np

ChainMethod = Literal["parallel", "sequential"]
JitMethod = Literal["jit", "native", "none"]
PotentialFunc = Callable[[np.ndarray], np.ndarray]
