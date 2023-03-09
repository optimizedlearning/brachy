import structure_util as su
from jax.tree_util import tree_map, Partial
from jax import numpy as jnp
import jax

import sys
sys.path.append('.')
import structure_util as su


def _cast_fp16(x):
    if x.dtype==jnp.float32:
        return x.astype(jnp.float16)
    else:
        return x

def _cast_fp32(x):
    if x.dtype==jnp.float16:
        return x.astype(jnp.float32)
    else:
        return x

def fp16_apply(apply):
    def new_apply(tree, global_config, *args, **kwargs):
        params_dtypes = tree_map(lambda x: x.dtype, tree['params'])
        buffers_dtypes = tree_map(lambda x: x.dtype, tree['buffers'])
        tree['params'] = tree_map(_cast_fp16, tree['params'])
        tree['buffers'] = tree_map(_cast_fp16, tree['buffers'])
        args = [tree_map(_cast_fp16, arg) for arg in args]
        kwargs = {k: tree_map(_cast_fp16, v) for k, v in kwargs.items()}
        
        state, value = apply(tree, global_config, *args, **kwargs)

        state['params'] = tree_map(lambda x, t: x.astype(t), state['params'], params_dtypes)
        state['buffers'] = tree_map(lambda x, t: x.astype(t), state['buffers'], buffers_dtypes)
        
        value = tree_map(_cast_fp32, value)

        return state, value
    return new_apply

def high_precision_apply(apply):
    def new_apply(tree, global_config, *args, **kwargs):
        args = tree_map(_cast_fp32, args)
        kwargs = tree_map(_cast_fp32, kwargs)
        
        state, value = apply(tree, global_config, *args, **kwargs)
        
        value = tree_map(_cast_fp32, value)

        return state, value
    return new_apply
            

def cast_node(node, path):
    node = su.copy_to_leaf(node)
    node['aux']['mixed_precision'] = {
        'old_apply': node['apply']
    }
    if 'force_high_precision' in node['aux'] and node['aux']['force_high_precision']:
        node['apply'] = high_precision_apply(node['apply'])
        return node

    node['apply'] = fp16_apply(node['apply'])

    return node

def cast_tree_f16(tree):
    mixed_precision_buffers = tree['buffers']['mixed_precision']
    del tree['buffers']['mixed_precision']

    half_tree = su.structure_tree_map(cast_node, tree)
    half_tree['buffers']['mixed_precision'] = mixed_precision_buffers
    return half_tree
        
def cast_back(tree):
    half_params_buffers, rest = su.split_tree(tree, [['params', 'buffers'], ['aux', 'apply']])
    mixed_precision_buffers = half_params_buffers['buffers']['mixed_precision']
    del half_params_buffers['buffers']['mixed_precision']
    types = mixed_precision_buffers['types']

    def cast(x, t):
        return x.astype(t.dtype)

    params_buffers = tree_map(cast, half_params_buffers, types)
    params_buffers['buffers']['mixed_precision'] = mixed_precision_buffers
    return su.merge_trees(rest, params_buffers)



def add_mixed_precision(float_tree, loss_scalar=1.0, output_type=jnp.float32):

    root_apply = float_tree['apply']
    half_tree = su.structure_tree_map(cast_node, float_tree)

    half_tree['buffers']['mixed_precision'] = {
        'loss_scalar': jnp.array(loss_scalar, dtype=jnp.float16),
        'output_type': jnp.ones(1, dtype=output_type),
    }

    return half_tree
