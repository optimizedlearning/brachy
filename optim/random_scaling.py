import structure_util as su
from jax.tree_util import tree_map, Partial
from jax import numpy as jnp
import jax

import sys
sys.path.append('.')
import rng_util

def randomly_scale(model_state, opt_state, opt_apply, distribution=jax.random.uniform, rng=None):
    state = {
        'subopt_state': opt_state,
        'true_iterate': model_state,
        'rng': rng
    }

    return state, Partial(randomly_scale_apply, subopt_apply=opt_apply, distribution=distibution)

def randomly_scale_apply(
    opt_state,
    model_state,
    value_and_grad_fn,
    distribution,
    subopt_apply,
    *args,
    **kwargs
    params_filter=su.get_params,
    params_merger=su.merge_trees):

    prev_true_iterate = opt_state['true_iterate']

    opt_state['subopt_state'], opt_state['true_iterate'], *value = subopt_apply(model_state, *args, **kwargs)

    opt_state['rng'], subkey = jax.random.split(opt_state['rng'])

    scale = distribution(subkey)

    params, rest = params_filter(opt_state['true_iterate'])
    prev_params = params_filter(prev_true_iterate)

    params = tree_map(
        lambda cur, prev: prev * scale  + cur * (1-scale),
        params,
        prev_params
    )

    model_state = params_merger(rest, params)

    return opt_state, model_state , *value
