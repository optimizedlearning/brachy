

import jax
from  jax import numpy as jnp


# eventually it might be cool to implement SGD as a structure tree
# but right now I want to get resnet working.

def SGD(params, grads, lr):
    return tree_map(lambda p, g: p-lr*g, params, grads)

