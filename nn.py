import jax
import numpy as np
from jax import numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

import rng_util

from jax.tree_util import Partial

import einops

import pprint
import gc

from state_util import ungroup_state, group_state_list

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

        att = softmax(logits, axis=-1, where=broadcast_mask) # [B, N, T, T] -> [B, N, T, T]
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



# the jax.nn.softmax function has some instabilities related to the where argument.
# This one doesn't
def softmax(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None) -> Array:
  r"""Softmax function.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Args:
    x : input array
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.
      Either an integer or a tuple of integers.
    where: Elements to include in the :code:`softmax`.
  """
  x_max = jnp.max(x, axis, where=where, initial=-jnp.inf, keepdims=True)
  unnormalized = jnp.exp(x - lax.stop_gradient(x_max))
  if where is not None:
    unnormalized = jnp.where(where, unnormalized, 0.0)
  return unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)

def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy between sets of logits and integer labels.
    Measures the probability error in discrete classification tasks in which
    the classes are mutually exclusive (each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.
    References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)
    Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Integers specifying the correct class for each input, with shape
        `[...]`.
    Returns:
    Cross entropy between each prediction and the corresponding target
    distributions, with shape `[...]`.
    """
    # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
    # we avoid subtracting the normalizer from all values, just from the values
    # for the correct labels.


    no_ignore = jax.lax.stop_gradient(labels!=-100)
    logits_max = jnp.max(logits, axis=-1, keepdims=True, where=no_ignore, initial=0.0)
    logits = logits - jax.lax.stop_gradient(logits_max)


    log_normalizer = jnp.log(jnp.sum(jnp.exp(logits), axis=-1, where=no_ignore))

    ignore_labels = jnp.where(no_ignore, labels, jnp.zeros_like(labels))
    label_logits = jnp.take_along_axis(logits, ignore_labels[..., None], axis=-1)[..., 0]

    return jnp.mean(log_normalizers - label_logits, axis=-1, where=no_ignore)

    # # ignore_labels = jnp.where(no_ignore, labels, jnp.zeros_like(labels))

    # # total = jax.lax.stop_gradient(jnp.sum(no_ignore))

    # label_logits = jnp.take_along_axis(logits, ignore_labels[..., None], axis=-1)[..., 0]

    # # log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    # # return jnp.sum(jnp.where(no_ignore, log_normalizers - label_logits, jnp.zeros_like(labels)))/total
