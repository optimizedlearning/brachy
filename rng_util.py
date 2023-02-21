
# listen, I'm not enough of SWE to be confident this *isn't* the horrible
# idea I think it probably is.

import jax
from jax.tree_util import Partial

_RNG = None

def __getattr__(name, value):
    def wrap(*args, **kwargs):
        rng = split()
        return getattr(jax.random, name)(rng, *args, **kwargs)
    return wrap


# copied from pytorch
def _calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_uniform(shape, gain=1.0, rng=None):
    if rng is None:
        rng = split()

    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)

    return gain / jnp.sqrt(fan_in + fan_out) * jax.random.uniform(rng, shape=shape, min_val=-1, max_val=1)

def init_rng(i: int) -> None:
    set_rng(jax.random.PRNGKey(i))

def set_rng(k: jax.random.KeyArray) -> None:
    global _RNG
    _RNG = k

# should we maintain the same 
# `num`` argument usage as jax.random.split?
# then the default num should be 2...
def split(num: int=1) -> jax.random.KeyArray:
    global _RNG
    next_RNG, rest = jax.random.split(_RNG, num + 1)
    _RNG = next_RNG
    return rest


def fold_in(data: int) -> None:
    global _RNG
    _RNG = jax.random.fold_in(_RNG, data)


class RNGState:
    def __init__(self, key: jax.random.KeyArray):
        self.start_key = key

    def __enter__(self):
        self.old_key = _RNG
        set_rng(self.start_key)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_rng(self.old_key)

