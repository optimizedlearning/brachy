
# listen, this whole concept of this module seems a bit suspect, but I'm not
# enough of SWE to be confident that it *isn't* a Bad Idea.
'''
This module's main purpose is to provide the context manage RNGState. What is it for? Consider the following code in pytorch:
S = torch.nn.Sequential(*[torch.nn.Linear(10,10) for _ in range(N)])

This initializes a list of N fully-connected layers chained together. Each of the N layers is intialized randomly and (pseudo)-independently.

In JAX, we need to keep explicit track of the pseudorandom state. Thus, instead we'd have to do something like:
chain = []
for _ in range(N):
    rng, subkey = jax.random.split(rng)
    chain.append(hax.nn.Linear(10, 10, rng))

S = hax.nn.Sequential(*chain)

This is kind of a pain, and if you forget to do the split at some point everything will be sad.
So, instead, the RNGState context manager allows the following:
with RNGState(rng):
    S = hax.nn.Sequential(*[hax.nn.Linear(10, 10) for _ in range(N)])

So, still slightly more annoying than pytorch, but maybe not too bad.

How does it work? When we have activated the context manarge RNGState(rng), the provided rng is saved in rng_util._RNG.
Then, to get a new rng we can call rng_util.split(), which will split rng_util._RNG, save one split subkey as the new value of
rng_util._RNG, and return the rest to the user.

Inside the module in hax.nn, if we see that the provided rng is the default value of None, then we will set rng = rng_util.split(). Thus, the two-line
version is equivalent to the multi-line version.


We also use this module to wrap all of jax.random. So, if you call p = rng_util.bernoulli(blah), it will be the same as:
rng = rng_util.split()
p = jax.random.bernoulli(rng, blah)
'''

import jax
from jax.tree_util import Partial

_RNG = None


def init_rng(i: int) -> None:
    set_rng(jax.random.PRNGKey(i))

def set_rng(k: jax.random.KeyArray) -> None:
    global _RNG
    _RNG = k

def get_rng() -> jax.random.KeyArray:
    return _RNG

# should we maintain the same 
# `num`` argument usage as jax.random.split?
# then the default num should be 2...
# But it seems better to have the num be the number of returned values
def split(num: int=1) -> jax.random.KeyArray:
    global _RNG
    next_RNG, rest = jax.random.split(_RNG, num + 1)
    _RNG = next_RNG
    return rest

class RNGState:
    def __init__(self, key: jax.random.KeyArray):
        self.start_key = key

    def __enter__(self):
        self.old_key = _RNG
        set_rng(self.start_key)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_rng(self.old_key)

    def get_rng(self):
        return _RNG


def fold_in(data: int) -> None:
    global _RNG
    _RNG = jax.random.fold_in(_RNG, data)


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

