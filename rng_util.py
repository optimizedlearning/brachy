
# listen, I'm not enough of SWE to be confident this *isn't* the horrible
# idea I think it probably is.

import jax

_RNG = None


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

