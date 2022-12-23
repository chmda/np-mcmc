# :game_die: NP-MCMC :game_die:

A python package for implementing Markov Chain Monte Carlo (MCMC) algorithms. It supports [Numba](https://numba.pydata.org/) jitted functions.

## :floppy_disk: Installation :floppy_disk:

<!--
To install the package, use pip:
```
pip install np-mcmc
```
-->

**TODO**

## :rocket: Usage :rocket:

To use the package, import it, create a kernel and create a MCMC object with this kernel. Then do warmup and run it to get samples.

```python
from numba import njit
import numpy as np
from np_mcmc import MCMC
from np_mcmc.kernels import MetropolisHastings

d: int = 5
mu: np.ndarray = np.random.normal(size=d)
inv_cov: np.ndarray = np.eye(d)


def log_density(
    x: np.ndarray, loc: np.ndarray = mu, inv_cov: np.ndarray = inv_cov
) -> np.ndarray:
    delta: np.ndarray = x - loc
    m: np.ndarray = delta.T @ inv_cov @ delta
    return -m / 2


log_normal_density = njit(log_density)

# Metropolis Hastings
step_size: float = 1.0
kernel = MetropolisHastings(log_normal_density, step_size=step_size)
mcmc = MCMC(kernel, num_warmup=10000)
mcmc.warmup(init_params=(np.zeros((d,)),))
mcmc.run(num_samples=10000)

samples: np.ndarray = mcmc.get_samples()
```

You can find more examples in [examples](./examples/).

## :scroll: License :scroll:

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
