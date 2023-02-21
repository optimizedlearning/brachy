import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

import rng_util

from jax.tree_util import Partial

import einops

import pprint
import gc

from state_util import ungroup_state, group_state_list

from types import SimpleNamespace

import functools

def merge_configs(*configs):
    ret = {}
    for config in configs:
        ret.update(config)

    return ret


STATE_ORGANIZER_RESERVED = [
    '_state',
    '_apply_fns',
    'local_config',
    '_global_config'
]




class StateOrganizer:

    def __init__(
        self,
        state=None,
        global_config=None,
        local_config=None,
        apply_fns=None,
        ):
        if state is None:
            state = {
                'params': {},
                'constants': {}
            }
        if apply_fns is None:
            apply_fns = {}
        
        if local_config is None:
            local_config = {}

        if global_config is None:
            global_config = {}

        self._state = state
        self._apply_fns = apply_fns
        self.local_config = local_config # we'll let users access local_config easily
        self._global_config = global_config # it's possible to screw something up by incorrectly accesssing this, so we make it harder.

    def update_global_config(self, update):
        self._global_config.update(update)

    def create_module(self, apply):
        pack = functools.partial(
            StateOrganizer,
            local_config=self.local_config,
            apply_fns=self._apply_fns)
        return Partial(apply, pack), self._state, self._global_config

    def get_state(self):
        return self._state

    def get_apply_fns(self):
        return self._apply_fns

    def __getattribute__(self, name):
        if name in STATE_ORGANIZER_RESERVED:
            return super().__getattribute__(name)


        if name in self._apply_fns:
            apply_fns = self._apply_fns
            params = self._state['params']
            constants = self._state['constants']
            global_config = self._global_config
            state = self.get_state()
            def apply(*args, **kwargs):
                x, next_state = apply_fns[name](
                    {
                        'params': params[name],
                        'constants': constants[name]
                    },
                    global_config,
                    *args, 
                    **kwargs)

                params[name] = next_state['params']
                constants[name] = next_state['constants']
                return x
            return apply
        
        if name in self._state['params']:
            return self._state['params'][name]
        if name in self._state['constants']:
            return self._state['constants'][name]

        return super().__getattribute__(name)

    def register_parameter(self, name, value):
        assert name not in self._state['params'], f"cannot create submodule {name}: a pre-existing parameter already has this name!"
        assert name not in self._state['constants'], f"cannot create submodule {name}: a pre-existing constant already has this name!"

        self._state['params'][name] = value

    def register_buffer(self, name, value):
        assert name not in self._state['params'], f"cannot create submodule {name}: a pre-existing parameter already has this name!"
        assert name not in self._state['constants'], f"cannot create submodule {name}: a pre-existing constant already has this name!"

        self._state['constants'][name] = value

    def __setattr__(self, name, value):
        '''
        sets an attribute.
        We assume that value is EITHER a:
        1. tuple (state, apply)  where state is a pytree of state info  and apply is
            an apply function for another module.
        2. a state, where state is a pytree of state info.

        in either case, the state info is stored as a trainable parameter.
        To make a non-trainable parameter, you must use register_buffer, as in pytorch.å
        '''
        if name in STATE_ORGANIZER_RESERVED:
            return super().__setattr__(name, value)

        if name in self._state['constants']:
            self.state['constants'][name] = value
            return super().__setattr__(name, value)  

        # try to unpack:
        if isinstance(value, tuple) and len(value) == 3:
            apply, state, global_config = value
            if not callable(apply):
                # this is some weird tuple parameter assignment I guess.
                # maybe we should just forbid such behavior, but anyway...
                state = value
        else:
            state = value
            apply = None
        
        if isinstance(state, dict):
            self._state['params'][name] = state['params']
            self._state['constants'][name] = state['constants']
        else:
            self._state['params'][name] = state

        if apply is not None:
            self._apply_fns[name] = apply
            self.update_global_config(global_config)

        return super().__setattr__(name, value)        

def Identity(rng=None):

    state = {
        'params': {},
        'constants': {}
    }

    global_config = {}
    return Identity_apply, state, global_config

def Identity_apply(state, global_config, x):
    del global_config
    return x, state

def Linear(in_features, out_features, bias=True, dtype=None, rng=None):
    if rng is None:
        rng = rng_util.split()

    rng, subkey = jax.random.split(rng)

    w = jax.random.uniform(
        key=subkey,
        shape=(out_features, in_features),
        minval=-jnp.sqrt(1/in_features),
        maxval=jnp.sqrt(1/in_features),
        dtype=None
    )

    params = {
        'weight': w
    }

    if bias:
        rng, subkey = jax.random.split(rng)
        b = jax.random.uniform(
            key=subkey,
            shape=(out_features,),
            minval=-jnp.sqrt(1/in_features),
            maxval=jnp.sqrt(1/in_features)
        )
        params['bias'] = b

    state = {
        'params': params,
        'constants': {},
    }
    global_config = {}
    return Linear_apply, state, global_config


def Linear_apply(state, global_config, x):
    del global_config
    params = state['params']

    weight = params['weight'].transpose()


    r = jnp.matmul(x, weight)

    if 'bias' in params:
        bias = params['bias']
        r = r + bias

    return r,  state




def Embedding(num_embeddings, embedding_dim, dtype=None, rng=None):
    if rng is None:
        rng = rng_util.split()

    rng, subkey = jax.random.split(rng)
    weight = jax.random.normal(
        key=subkey,
        shape=(num_embeddings, embedding_dim),
        dtype=dtype
    )

    params = {
        'weight': weight
    }

    state = {
        'params': params,
        'constants': {},
    }

    global_config = {}
    return Embedding_apply, state, global_config


def Embedding_apply(state, global_config, idx):
    del global_config
    weight = state['params']['weight']
    return weight[idx, :], state


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
    


    applies = [s_a[0] for s_a in submodules]
    states = [s_a[1] for s_a in submodules]
    configs = [s_a[2] for s_a in submodules]

    seq_state = group_state_list(states)
    apply_fn = Partial(Sequential_apply, applies)

    global_config = merge_configs(*configs)

    return apply_fn, seq_state, global_config


def Sequential_apply(applies, state, global_config, x):
    states = ungroup_state(state)

    next_states = []

    for s, apply in zip(states, applies):
        x, state_update = apply(s, global_config, x)
        next_states.append(state_update)

    return x, group_state_list(next_states)


def LayerNorm(normalized_shape, eps=1e-05, rng=None):

    organizer = StateOrganizer()

    organizer.weight = jnp.ones(normalized_shape)

    organizer.bias = jnp.zeros(normalized_shape)

    organizer.register_buffer('eps', eps)

    return organizer.create_module(Layernorm_apply)




def Layernorm_apply(pack_state, state, global_config, x):
    module = pack_state(state, global_config)

    e_x = jnp.average(x, axis=-1, keepdims=True)
    v_x = jnp.average((x-e_x)**2, axis=-1, keepdims=True)
    

    ln = (x - e_x)/jnp.sqrt(v_x + module.eps) * module.weight + module.bias

    return ln, module.get_state()


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

    rng, subkey = jax.random.split(rng)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    state = {
        'params': {},
        'constants': {
            'padding': padding,
            'stride': stride,
            'dilation': dilation,
            'feature_group_count': groups,
        }
    }

    k = groups / (in_channels * kernel_size[0] * kernel_size[1])

    state['params']['weight'] = jax.random.uniform(
        key=subkey,
        shape=(out_channels, in_channels//groups, kernel_size[0], kernel_size[1]),
        minval=-jnp.sqrt(k),
        maxval=jnp.sqrt(k)
    )

    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding))

    if bias:
        rng, subkey = jax.random.split(rng)
        state['params']['bias'] = jax.random.uniform(
            key=subkey,
            shape=(out_channels,),
            minval=-jnp.sqrt(k),
            maxval=jnp.sqrt(k),
        )

    global_config = {}
    return Conv2d_apply, state, global_config

    
def Conv2d_apply(state, global_config, x):
    '''
    perform a convolution.

    arguments:
        state: a state pytree. 

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
    
    weight = state['params']['weight']

    constants = SimpleNamespace(**state['constants'])
    

    


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



    if 'bias' in state['params']:
        bias = state['params']['bias']
        
        conv = conv + einops.rearrange(bias, '(N C H W) -> N C H W', N=1, H=1, W=1)

    return conv, state


def Dropout(prob_zero=0.5, rng=None):
    if rng is None:
        rng = rng_util.split()

    state = {
        'params': {},
        'constants': {
            'rng': rng,
            'prob_zero': prob_zero
        }
    }
    global_config = {
        'train_mode': True
    }
    return Dropout_apply, state, global_config

def Dropout_apply(state, global_config, x):
    '''
    we will allow x to be a pytree for more generality, although
    that does make the implementation a bit more opaque
    '''
    if not global_config['train_mode']:
        return x, state

    rng = state['constants']['rng']
    prob_zero = state['constants']['prob_zero']

    prob_one = 1.0 - prob_zero

    x_flat, treedef = tree_flatten(x)

    rng, *subkeys = jax.random.split(rng, len(x_flat)+1)

    dropout_flat = [v * jax.random.bernoulli(k, prob_one)/prob_one for v, k in zip(x_flat, subkeys)]

    x_dropout = tree_unflatten(treedef, dropout_flat)

    next_state = {
        'params': {},
        'constants': {
            'rng': rng,
            'prob_zero': prob_zero
        }
    }

    return x_dropout, next_state



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

    

    organizer = StateOrganizer(local_config={'num_heads': num_heads})

    # the pytorch implementation is full of random special cases.
    # Let's try to not do that here. This requires one special case
    # parameter extraction here, and then none later one.

    with rng_util.RNGState(rng):

        organizer.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        organizer.k_proj = Linear(k_dim, embed_dim, bias=bias)
        organizer.v_proj = Linear(v_dim, embed_dim, bias=bias)

    
    return organizer.create_module(MultiheadAttention_apply)

def MultiheadAttention_apply(pack_state, state, global_config, q, k, v, mask=None):
    # q is [B, T, C]
    # k is [B, T, K]
    # v is [B, T, V]
    # mask is is an array of booleans of shape
    # [b, n, L, L]
    # where b is either 1 or B
    # n is either 1 or num_heads
    # L is at least T.

    *_, T, C = q.shape

    module = pack_state(state, global_config)


    num_heads = module.local_config['num_heads']

    q = module.q_proj(x)
    k = module.k_proj(x)
    v = module.v_proj(x)

    # q, k, v all are all [B, T, C]

    q = einops.rearrange(q, 'b t (n h) -> b, n, t, h', n=num_heads) # [B T C] -> [B N T H]
    k = einops.rearrange(k, 'b t (n h) -> b, n, t, h', n=num_heads)
    v = einops.rearrange(m, 'b t (n h) -> b, n, t, h', n=num_heads)

    logits = einops.einsum(q, k, 'b n t1 c, b n t2 c -> b n t1 t2') # [B, N, T, H] x [B, N, T, H] -> [B, N, T, T]


    if mask is not  None:
        broadcast_mask = jnp.broadcast_to(mask[:, :, :T, :T], logits.shape)
        masked_logits = jnp.where(broadcast_mask, logits, -jnp.inf)

        att = jax.nn.softmax(masked_logits, axis=-1) # [B, N, T, T] -> [B, N, T, T]
    else:
        att = jax.nn.softmax(logits, axis=-1)

    values = einops.einsum(att, v, 'b n t1 t2, b n t2 h -> b n t1 h') # [B, N, T, T] x [B, N, T, H] -> [B, N, T, H]
    values = einops.rearrange(values, 'b n t h -> b t (n h)') # [B N T H] -> [B T C]

    return values, state

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


    organizer.proj = Linear(embed_dim, embed_dim, bias=bias, rng=rng)

    organizer = StateOrganizer()

    organizer.MHA = MultiheadAttention(
        embed_dim,
        num_heads,
        bias,
        rng=rng
        )

    return organizer.create_module(CausalSelfAttention)

def CausalSelfAttention_apply(pack_state, state, global_config, x):

    module = pack_state(state, global_config)

    *_, T, C = x.shape

    # should we actually be  storing this as a constant? then we'd need to know the
    # T ahead of time (although I guess we could fall back to this case if needed... 
    # the conditional will be ok even with jax.jit since it depends on the shape)
    causal_mask = jnp.tril(jnp.ones((1, 1, T, T))) 

    return module.MHA(x, x, x, causal_mask)




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
