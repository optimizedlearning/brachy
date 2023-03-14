

from jax import numpy as jnp

from typing import overload, Any, Callable, Literal, Optional, Sequence, Tuple, Union

def cosine_annealing(t: int, T_max: int, eta_min: = 0.0, lr: float =1.0):
    return (1 + jnp.cos(jnp.pi * t / T_max)) * (lr- eta_min) * 0.5 + eta_min


def linear_warmup(t: int, warmup_max: int,  lr: float =1.0):
    return jnp.minimum(1.0, t/warmup_max) * lr

def linear_decay(t: int, T_max: int, lr: float =1.0):
    return lr * t/T_max
