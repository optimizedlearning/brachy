
import jax
from .static_wrapper import improved_static

from .sidecall import sidecall_wrapper, sidecall



jit = sidecall_wrapper(improved_static(jax.jit))
# jit = improved_static(jax.jit)