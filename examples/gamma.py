from numba import njit, jit
import numpy as np
import matplotlib.pyplot as plt
import time
from np_mcmc import MCMC
from np_mcmc.kernels import MALA

alpha: float = np.random.uniform(low=0.0, high=2.0)
beta: float = np.random.uniform(low=0.0, high=1.0)


def log_density(x: np.ndarray, alpha: float = alpha, beta: float = beta) -> np.ndarray:
    return (alpha - 1) * np.log(x) - beta * x


def grad_log_density(
    x: np.ndarray, alpha: float = alpha, beta: float = beta
) -> np.ndarray:
    return (alpha - 1) / x - beta


jit_log_density = njit(log_density)
jit_grad_log_density = njit(grad_log_density)

# MALA
step_size: float = 0.574
kernel = MALA(jit_log_density, jit_grad_log_density, step_size=step_size)
mcmc = MCMC(kernel, num_warmup=1000)

start_time = time.time()
mcmc.warmup(init_params=(np.ones((1,)),))
elapsed = time.time() - start_time
print("Warmup:", elapsed)

start_time = time.time()
mcmc.run(num_samples=10000)
elapsed = time.time() - start_time
print("Run:", elapsed)

samples: np.ndarray = mcmc.get_samples()
mu: np.ndarray = np.mean(samples, axis=0)
var: np.ndarray = np.var(samples, axis=0)
estimated_alpha: float = mu**2 / var
estimated_beta: float = mu / var

print("Original parameters:", "alpha = ", alpha, ";", "beta =", beta)
print(
    "Estimated parameters:", "alpha = ", estimated_alpha, ";", "beta =", estimated_beta
)

_, bins, _ = plt.hist(samples[:, 0], density=True, bins=100)
plt.plot(bins, np.exp(log_density(bins)))
plt.grid()
plt.show()
