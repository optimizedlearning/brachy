import jax
import numpy as np
from jax import numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

import rng_util

from jax.tree_util import Partial

import einops

import functional
import pprint
import gc


from types import SimpleNamespace

from jax._src.typing import Array, ArrayLike, DType, DTypeLike
from typing import overload, Any, Callable, Literal, Optional, Sequence, Tuple, Union

import functools

import structure_utils as su

def Identity():

    tree = su.empty_tree()
    tree['apply'] = Identity_apply

    global_config = {}

    return tree, global_config

def Identity_apply(tree, global_state, x):
    return tree, x

def Linear(in_features, out_features, bias=True, dtype=None, rng=None):
    if rng is None:
        rng = rng_util.split()

    if bias:
        rng, rng_bias = jax.random.split(rng)

    w = jax.random.uniform(
        key=rng,
        shape=(out_features, in_features),
        minval=-jnp.sqrt(1/in_features),
        maxval=jnp.sqrt(1/in_features),
        dtype=None
    )

    params = {
        'weight': w
    }

    if bias:
        b = jax.random.uniform(
            key=rng_bias,
            shape=(out_features,),
            minval=-jnp.sqrt(1/in_features),
            maxval=jnp.sqrt(1/in_features)
        )
        params['bias'] = b

    tree = {
        'params': params,
        'constants': {},
        'aux': {},
        'apply': Linear_apply,
        'submodules': {}
    }
    ### the above definition could instead be written as:
    # tree = su.fill_tree({
    #     'params': params,
    #     'apply': Linear_apply
    # })
    #
    # We leave in the more verbose way for pedagogical reasons.
    ####    
    
    global_config = {}

    return tree, global_config


def Linear_apply(tree, global_config, x):
    params = tree['params']

    weight = params['weight'].transpose()


    r = jnp.matmul(x, weight)

    if 'bias' in params:
        bias = params['bias']
        r = r + bias

    # technically only the 'params' and 'constants' keys in the returned
    # tree (and its submodules) are important. The others will be ignored.
    # So, we could instead return the value su.filter_keys(tree, ['params', 'constants']).
    # But that is more writing.
    return tree, r


def Embedding(num_embeddings, embedding_dim, dtype=None, rng=None):
    if rng is None:
        rng = rng_util.split()

    weight = jax.random.normal(
        key=rng,
        shape=(num_embeddings, embedding_dim),
        dtype=dtype
    )

    params = {
        'weight': weight
    }

    tree = su.fill_tree({
        'params': params,
        'apply': Embedding_apply
    })

    global_config = {}

    return tree, global_config


def Embedding_apply(tree, global_config, idx):
    weight = tree['params']['weight']
    return tree, weight[idx, :]


def module_from_func(func):
    def wrapped_func(tree, global_config, *args, **kwargs):
        return tree, func(*args, **kwargs)
    tree = su.fill_tree({'apply': wrapped_func})
    return tree, {}

def Sequential(*submodules, rng=None):
    '''
    chains together a list of state/apply_fn pairs ala torch.nn.Sequential

    Each submodule chained together like this must take as input one pytree
    and return one pytree. No multiple arguments please for now.
    
    arguments:
        submodules: An iterable of (state, apply_fn, global_config) tuples
            where each `state` is a pytree and each `apply_fn` is a function whose first
            argument is pytree of the same shape as the corresponding `state`. 

        return_torch: if True, return a pytorch Sequential module in addition to the
            Hax sequential information.

    returns:
        seq_state, apply_fn, and possibly also t_state, t_module.
        
    '''


    if len(submodules) == 0:
        raise ValueError(f"You must provide a non-empty list to Sequential!")
    

    tree = su.empty_tree()
    
    for i, s in enumerate(submodules):
        tree['submodules'][i] = s[0]

    tree['apply'] = Sequential_apply

    global_config = su.merge_configs(*[s[1] for s in submodules])

    return tree, global_config


def Sequential_apply(tree, global_config, x):
    next_tree = su.copy_dict(tree)

    for i in range(len(su.children(next_tree))):
        submodule = next_tree['submodules'][i]

        next_params_consts, x = submodule['apply'](submodule, global_config, x)
        next_tree['submodules'][i] = su.merge_trees(submodule, next_params_consts)

    return next_tree, x


def LayerNorm(normalized_shape, eps=1e-05, rng=None):

    organizer = su.StateOrganizer()

    organizer.weight = jnp.ones(normalized_shape)

    organizer.bias = jnp.zeros(normalized_shape)

    organizer.register_constant('eps', eps)

    return organizer.create_module(Layernorm_apply)




def Layernorm_apply(tree, global_config, x):
    module = su.StateOrganizer(tree, global_config)

    e_x = jnp.average(x, axis=-1, keepdims=True)
    v_x = jnp.average((x-e_x)**2, axis=-1, keepdims=True)
    

    ln = (x - e_x)/jnp.sqrt(v_x + module.eps) * module.weight + module.bias

    return module.get_state_update(), ln


def Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros',
    dtype=None,
    return_torch=False,
    rng=None):
    '''
    See the torch.nn.Conv2d description for what the arguments are.
    '''
    assert padding_mode=='zeros', "currently only the 'zeros' padding_mode is supported, sorry!"
    if rng is None:
        rng = rng_util.split()

    if bias:
        rng, bias_rng = jax.random.split(rng)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding))


    tree = su.fill_tree({
        'params': {},
        'constants': {
            'padding': padding,
            'stride': stride,
            'dilation': dilation,
            'feature_group_count': groups,
        },
        'apply': Conv2d_apply,
    })

    k = groups / (in_channels * kernel_size[0] * kernel_size[1])

    tree['params']['weight'] = jax.random.uniform(
        key=rng,
        shape=(out_channels, in_channels//groups, kernel_size[0], kernel_size[1]),
        minval=-jnp.sqrt(k),
        maxval=jnp.sqrt(k)
    )

    if bias:
        tree['params']['bias'] = jax.random.uniform(
            key=bias_rng,
            shape=(out_channels,),
            minval=-jnp.sqrt(k),
            maxval=jnp.sqrt(k),
        )

    global_config = {}
    return tree, global_config
    
def Conv2d_apply(tree, global_config, x):
    '''
    perform a convolution.

    arguments:
        tree: a structure tree

        x: a shape [N, Cin, Hin, Win] tensor, where N is usually batch dimension,
            C is channels and ... represents an arbitrary number of
            shape dimension (usually 2, sometimes 1, occasionally 3, but could
            be anything)

            NOTE: pytorch allows x to have shape [Cin, Hin, Win]. Currently this
            will thrown an error here. To be fixed late (maybe).
    
    returns:
        conv: a shape [N, Cout, Hout, Wout] tensor where Cout is the 
            number of output channels.
            The size of the shape dimensions Hout, Wout 
            will depend on potential padding of the convolution operation.
    '''
    
    weight = tree['params']['weight']

    constants = SimpleNamespace(**tree['constants'])
    

    


    conv = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=constants.stride,
        padding=constants.padding,
        lhs_dilation=None,
        rhs_dilation=constants.dilation,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        feature_group_count=constants.feature_group_count,
        batch_group_count=1,
        precision=None,
        preferred_element_type=None)



    if 'bias' in tree['params']:
        bias = tree['params']['bias']
        
        conv = conv + einops.rearrange(bias, '(N C H W) -> N C H W', N=1, H=1, W=1)

    return tree, conv


def Dropout(prob_zero=0.5, rng=None):
    if rng is None:
        rng = rng_util.split()

    tree = su.fill_tree({
        'params': {},
        'constants': {
            'rng': rng,
            'prob_zero': prob_zero
        },
        'apply': Dropout_apply,
    })
    global_config = {
        'train_mode': True
    }
    return tree, global_config

def Dropout_apply(tree, global_config, x):
    '''
    we will allow x to be a pytree for more generality, although
    that does make the implementation a bit more opaque
    '''
    if not global_config['train_mode']:
        return tree, x

    next_tree = su.copy_dict(tree)

    rng = next_tree['constants']['rng']
    prob_zero = next_tree['constants']['prob_zero']

    prob_one = 1.0 - prob_zero

    x_flat, treedef = tree_flatten(x)

    rng, *subkeys = jax.random.split(rng, len(x_flat)+1)

    dropout_flat = [v * jax.random.bernoulli(k, prob_one, shape=v.shape)/prob_one for v, k in zip(x_flat, subkeys)]

    x_dropout = tree_unflatten(treedef, dropout_flat)

    next_tree['constants']['rng'] = rng

    return next_tree, x_dropout



# ok, I wanted this to be the same as torch, but my god their implemention of
# multihead attention is way over-complicated. So, we opt for readability here
# instead.
def MultiheadAttention(
    embed_dim,
    num_heads,
    bias=True,
    k_dim=None,
    v_dim=None,
    rng=None):
    '''
    cls: class object
    return_torch:  whether to return a pytorch object.
    '''
    if k_dim is None:
        k_dim = embed_dim
    if v_dim is None:
        v_dim = embed_dim

    if rng is None:
        rng = rng_utils.split()

    

    organizer = su.StateOrganizer()

    organizer.register_aux('num_heads', num_heads)

    # the pytorch implementation is full of random special cases.
    # Let's try to not do that here. This requires one special case
    # parameter extraction here, and then none later one.

    with rng_util.RNGState(rng):

        organizer.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        organizer.k_proj = Linear(k_dim, embed_dim, bias=bias)
        organizer.v_proj = Linear(v_dim, embed_dim, bias=bias)

    
    return organizer.create_module(MultiheadAttention_apply)

def MultiheadAttention_apply(tree, global_config, q, k, v, mask=None):
    # q is [B, T, C]
    # k is [B, T, K]
    # v is [B, T, V]
    # mask is is an array of booleans of shape
    # [b, n, L, L]
    # where b is either 1 or B
    # n is either 1 or num_heads
    # L is at least T.



    module = su.StateOrganizer(tree, global_config)


    num_heads = module.num_heads

    *_, T, C = q.shape
    H = C / num_heads 

    q = module.q_proj(q)
    k = module.k_proj(k)
    v = module.v_proj(v)

    # q, k, v all are all [B, T, C]

    q = einops.rearrange(q, 'b t (n h) -> b n t h', n=num_heads) # [B T C] -> [B N T H]
    k = einops.rearrange(k, 'b t (n h) -> b n t h', n=num_heads)
    v = einops.rearrange(v, 'b t (n h) -> b n t h', n=num_heads)

    logits = einops.einsum(q, k, 'b n t1 h, b n t2 h -> b n t1 t2') # [B, N, T, H] x [B, N, T, H] -> [B, N, T, T]
    logits = logits / jnp.sqrt(H)



    if mask is not  None:
        broadcast_mask = jnp.broadcast_to(mask[:, :, :T, :T], logits.shape)
        masked_logits = jnp.where(broadcast_mask, logits, 0.0)

        # x_max = jnp.max(logits, axis=-1, where=broadcast_mask , initial=-jnp.inf, keepdims=True)
        # unnormalized = jnp.exp(logits - jax.lax.stop_gradient(x_max))
        # unnormalized = jnp.where(broadcast_mask, unnormalized, 0.0)
        # att = unnormalized / jnp.sum(unnormalized, axis=-1, where=broadcast_mask, keepdims=True)

        att = functional.softmax(logits, axis=-1, where=broadcast_mask) # [B, N, T, T] -> [B, N, T, T]
    else:
        att = jax.nn.softmax(logits, axis=-1)

    values = einops.einsum(att, v, 'b n t1 t2, b n t2 h -> b n t1 h') # [B, N, T, T] x [B, N, T, H] -> [B, N, T, H]
    values = einops.rearrange(values, 'b n t h -> b t (n h)') # [B N T H] -> [B T C]

    return module.get_state_update(), values

def CausalSelfAttention(
    embed_dim,
    num_heads,
    bias=True,
    rng=None):
    '''
    cls: class object
    return_torch:  whether to return a pytorch object.
    '''
    if rng is None:
        rng = rng_utils.split()


    organizer = su.StateOrganizer()

    organizer.MHA = MultiheadAttention(
        embed_dim,
        num_heads,
        bias,
        rng=rng
        )

    return organizer.create_module(CausalSelfAttention_apply)

def CausalSelfAttention_apply(tree, global_config, x):

    module = su.StateOrganizer(tree, global_config)

    *_, T, C = x.shape

    # should we be storing this as a constant instead? then we'd need to know the
    # T ahead of time (although I guess we could fall back to this case if needed... 
    # A conditional will be ok even with jax.jit since it depends on the shape)
    causal_mask = jnp.tri(T, k=0).reshape((1, 1, T, T))

    return module.get_state_update(), module.MHA(x, x, x, causal_mask)


def BatchNorm(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, axis=None, batch_axis=None):
    organizer = su.StateOrganizer(global_config={'train_mode': True, 'batch_axis': batch_axis})

    if affine:
        organizer.register_parameter('scale', jnp.ones(num_features))
        organizer.register_parameter('bias', jnp.zeros(num_features))

    organizer.register_constant('eps', eps)
    organizer.register_constant('momentum', momentum)

    if momentum is None:
        organizer.register_constant('num_batches_tracked', 0)
    
    if track_running_stats:
        organizer.register_constant('running_mean', jnp.zeros(num_features))
        organizer.register_constant('running_var', jnp.ones(num_features))

    organizer.register_aux('affine', affine)
    organizer.register_aux('track_running_stats', track_running_stats)
    organizer.register_aux('axis', axis)

    return organizer.create_module(BatchNorm_apply)

def BatchNorm_apply(tree, global_config, x):
    #
    # pytorch is annoying here: the actual normalization in batch norm is done using
    # the UNBIASED variance estimate. However, the running variance statistic is calculated
    # using the BIASED variance estimate (this is in contradiction with their docs).
    # I'd rather use the same kind of estimate for both, but let's just try to be consistent
    # with pytorch for now.
    #
    organizer = su.StateOrganizer(tree, global_config)

    batch_axis = global_config['batch_axis']

    axis = organizer.axis
    if axis is None:
        if batch_axis is not None:
            axis = 0
        else:
            axis = 1



    if organizer.momentum is None:
        organizer.num_batches_tracked = organizer.num_batches_tracked + 1
        momentum = 1.0/organizer.num_batches_tracked
    else:
        momentum = organizer.momentum
    
    use_running_stats = (not global_config['train_mode']) and organizer.track_running_stats


    broadcast_shape = [1]* x.ndim
    broadcast_shape[axis] = x.shape[axis]
    broadcast_shape = tuple(broadcast_shape)


    if use_running_stats:
        y = (x - organizer.running_mean.reshape(broadcast_shape))/jnp.sqrt(organizer.running_var.reshape(broadcast_shape))

    if not use_running_stats:
        stats_axes = list(range(x.ndim))
        stats_axes.pop(axis)
        stats_axes = tuple(stats_axes)



        mean = jnp.mean(x, axis=stats_axes, keepdims=True)
        second_moment = jnp.mean(x**2, axis = stats_axes, keepdims=True)

        if batch_axis is not None:
            if isinstance(batch_axis, str):
                batch_axis = [batch_axis]

            for ba in batch_axis:
                mean = jax.lax.pmean(mean, axis_name=ba)
                second_moment = jax.lax.pmean(second_moment, axis_name=ba)


        var = second_moment - mean**2


        y = (x - mean)/jnp.sqrt(var)

        if organizer.track_running_stats:
            buffer_shape = organizer.running_mean.shape


            if batch_axis is None:
                # I can't figure out how to get ahold of the size of the batch axis when it is pmapped or vmapped.
                # Probably there is a way, but really you shouldn't be pmapping or vmapping something so small that
                # not doing this multiply will make any difference.
                # And anyway, we are only doing this because pytorch has a weirdly inconsistent usage anyhow...
                num_to_avg = 1/x.shape[axis]
                for d in x.shape:
                    num_to_avg = num_to_avg * d
                var *= num_to_avg/(num_to_avg - 1)


            organizer.running_mean = (1 - momentum) * organizer.running_mean + momentum * mean.reshape(buffer_shape)
            organizer.running_var = (1 - momentum) * organizer.running_var + momentum * var.reshape(buffer_shape) 


    if organizer.affine:
        y =  y * organizer.scale.reshape(broadcast_shape) + organizer.bias.reshape(broadcast_shape)

    return organizer.get_state(), y

