from brachy import structure_util as su
from jax.tree_util import tree_map, Partial
from jax import numpy as jnp
from jax import custom_vjp
from jax import custom_jvp


import jax



def _cast_fp16(x):
    if jnp.array(x).dtype==jnp.float32:
        return jnp.array(x).astype(jnp.float16)
    else:
        return x

def _cast_fp32(x):
    if jnp.array(x).dtype==jnp.float16:
        return jnp.array(x).astype(jnp.float32)
    else:
        return x

def fp16_apply(apply):
    def new_apply(tree, global_config, *args, **kwargs):
        params_dtypes = tree_map(lambda x: jnp.array(x).dtype, tree['params'])
        buffers_dtypes = tree_map(lambda x: jnp.array(x).dtype, tree['buffers'])
        tree['params'] = tree_map(_cast_fp16, tree['params'])
        tree['buffers'] = tree_map(_cast_fp16, tree['buffers'])
        args = [tree_map(_cast_fp16, arg) for arg in args]
        kwargs = {k: tree_map(_cast_fp16, v) for k, v in kwargs.items()}
        
        state, value = apply(tree, global_config, *args, **kwargs)

        state['params'] = tree_map(lambda x, t: jnp.array(x).astype(t), state['params'], params_dtypes)
        state['buffers'] = tree_map(lambda x, t: jnp.array(x).astype(t), state['buffers'], buffers_dtypes)
        
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
    node['static']['mixed_precision'] = {
        'old_apply': node['apply']
    }
    if 'force_high_precision' in node['static'] and node['static']['force_high_precision']:
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
    half_params_buffers, rest = su.split_tree(tree, [['params', 'buffers'], ['static', 'apply']])
    mixed_precision_buffers = half_params_buffers['buffers']['mixed_precision']
    del half_params_buffers['buffers']['mixed_precision']
    types = mixed_precision_buffers['types']

    def cast(x, t):
        return x.astype(t.dtype)

    params_buffers = tree_map(cast, half_params_buffers, types)
    params_buffers['buffers']['mixed_precision'] = mixed_precision_buffers
    return su.merge_trees(rest, params_buffers)


@custom_vjp
def scale_in_backwards(x, s):
    return x

def scale_in_backwards_fwd(x, s):
    return scale_in_backwards(x, s), s

def scale_in_backwards_bwd(s, g):
    return g * s, 0.0

scale_in_backwards.defvjp(scale_in_backwards_fwd, scale_in_backwards_bwd)


# currently assumes first argument  of loss is the tree and second return value is the loss.
def mixed_precision_loss(loss):
    def mixed_loss(tree, *args, **kwargs):

        loss_scalar = tree['buffers']['mixed_precision']['loss_scalar']
        output_type = tree['static']['mixed_precision']['output_type']

        tree = su.map_params_buffers(lambda x: scale_in_backwards(x, 1.0/loss_scalar.astype(output_type)), tree)
        (tree, value, *rest) = loss(tree, *args, **kwargs)
        value =  scale_in_backwards(value, loss_scalar)
        return (tree, value, *rest)
    return mixed_loss


def mixed_precision_tree(tree_and_config, loss_scalar=1.0, output_type=jnp.float32):
    float_tree = tree_and_config[0]
    config = tree_and_config[1]

    root_apply = float_tree['apply']
    half_tree = su.structure_tree_map(cast_node, float_tree)

    half_tree['buffers']['mixed_precision'] = {
        'loss_scalar': jnp.array(loss_scalar, dtype=jnp.float16),
    }

    half_tree['static']['mixed_precision'] = {
        'output_type': output_type
    }

    config['mixed_precision'] = True

    return half_tree, config
