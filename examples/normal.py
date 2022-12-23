from numba import njit, jit
import numpy as np
import matplotlib.pyplot as plt
import time
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

start_time = time.time()
mcmc.warmup(init_params=(np.zeros((d,)),))
elapsed = time.time() - start_time
print("Warmup:", elapsed)

start_time = time.time()
mcmc.run(num_samples=10000)
elapsed = time.time() - start_time
print("Run:", elapsed)

samples: np.ndarray = mcmc.get_samples()
print("MCMC mean:", np.mean(samples, axis=0))
print("Original mean:", mu)

for j in range(samples.shape[1]):
    plt.plot(samples[:, j], label=f"Dim {j+1}")
plt.grid()
plt.legend()
plt.show()
