from brachy import structure_util as su
from jax.tree_util import tree_map, Partial
from jax import numpy as jnp
import jax

import sys
from brachy import  rng_util

def random_scale(optimizer, model_tree, distribution=jax.random.uniform, interpolate=True, params_filter=su.get_params, params_merger=su.merge_trees,rng=None):
    if rng is None:
        rng = rng_util.split()

    organizer = su.StateOrganizer()

    organizer.register_buffer('rng', rng)
    organizer.optimizer = optimizer
    organizer.register_aux('distribution', distribution)
    organizer.register_aux('params_filter', params_filter)
    organizer.register_aux('params_merger',  params_merger)

    params, rest = params_filter(model_tree)

    organizer.register_buffer('true_params',
        params
    )
    
    organizer.register_aux('interpolate', interpolate)

    return organizer.create_module(random_scale_apply)

def random_scale_apply(
    opt_state,
    opt_config,
    hparams,
    value_and_grad_fn,
    model_tree,
    *args,
    **kwargs):

    organizer = su.StateOrganizer(opt_state, opt_config)

    true_params = organizer.true_params

    base_optimizer = organizer.optimizer

    params, rest = organizer.params_filter(model_tree)

    (updated_model, *value) = organizer.optimizer(hparams, value_and_grad_fn, model_tree, *args, **kwargs)

    updated_params, updated_rest = organizer.params_filter(updated_model)

    offset = tree_map(lambda x, y: x-y, updated_params, params)

    organizer.rng, subkey = jax.random.split(organizer.rng)

    scale = organizer.distribution(subkey)

    if organizer.interpolate:
        true_scale = 1.0
    else:
        true_scale = scale

    organizer.true_params = tree_map(lambda o, t: true_scale * o + t, offset, true_params)

    scaled_params = tree_map(lambda o, t: scale * o + t, offset, true_params)

    updated_model = organizer.params_merger(scaled_params, updated_rest)

    return organizer.get_state(), updated_model , *value
