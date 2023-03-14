import jax
from jax import numpy as jnp
from jax.tree_util import tree_map

import sys
sys.path.append('.')
import structure_util as su


def _per_coordinate_clip(x, value):
    return jnp.clip(x, a_min=-value, a_max=value)


def clip_grads(optimizer, clip_value=1.0, clip_type='per_coordinate'):
    tree, config = optimizer
    clip_tree = {
        'params': {
            'clip_value': clip_value,
        },
        'buffers': {},
        'aux': {
            'clip_type': clip_type
        },
        'submodules': {
            'optimizer': tree
        },
        'apply': clip_apply
    }

    return clip_tree, config

def clip_apply(
    opt_tree,
    opt_config,
    value_and_grad_fn,
    *args,
    **kwargs):

    clip_type = opt_tree['aux']['clip_type']
    supported_clip_types = ['per_coordinate']
    if clip_type not in supported_clip_types:
        raise ValueError(f"unsupported clip_type: {clip_type}. Supported types are: {supported_clip_types}")
    if clip_type == 'per_coordinate':
        clip_fn = _per_coordinate_clip

    clip_value = opt_tree['params']['clip_value']

    def new_value_and_grad_fn(*vg_args, **vg_kwargs):
        out, grad = value_and_grad_fn(*vg_args, **vg_kwargs)
        grad  = tree_map(lambda g: clip_fn(g, clip_value), grad)
        return out, grad

    base_opt = opt_tree['submodules']['optimizer']
    base_update, *output = su.apply_tree(base_opt, opt_config, new_value_and_grad_fn, *args, **kwargs)

    update = dict(opt_tree)
    update['submodules']['optimizer'] = base_update

    return update, *output
