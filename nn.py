


import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_map

import rng_util

from jax.tree_util import Partial

import einops

import pprint
import gc

from state_util import ungroup_state, group_state_list

from types import SimpleNamespace

import functools

# import torch

# class return_torch_hack:
#     def __init__(self, value):
#         self.value = value

# RETURN_TORCH = return_torch_hack(True)


# class set_return_torch:
#     def __init__(self, value):
#         self.value = value
         
#     def __enter__(self):
#         self.prev_value = RETURN_TORCH
#         RETURN_TORCH.value = self.value
#         return self
     
#     def __exit__(self, exc_type, exc_value, exc_traceback):
#         RETURN_TORCH.value = self.prev_value


# def t_to_jnp(tensor):
#     return jnp.array(tensor.detach().numpy())



STATE_ORGANIZER_RESERVED = [
    '_state',
    '_apply_fns',
    'config'
]




class StateOrganizer:

    def __init__(
        self,
        state=None,
        config=None,
        apply_fns=None,
        ):
        if state is None:
            state = {
                'params': {},
                'constants': {}
            }
        if apply_fns is None:
            apply_fns = {}
        
        if config is None:
            config = SimpleNamespace()

        self._state = state
        self._apply_fns = apply_fns
        self.config = config

    def create_module(self, cls):
        pack = functools.partial(
            StateOrganizer,
            config=self.config,
            apply_fns=self._apply_fns)
        return self._state, Partial(cls.apply, pack)

    def get_state(self):
        return self._state
        # params = self._state['params']
        # constants = self._state['constants']

        # state = {
        #     'params': {}
        #     'constants': {}
        # }

        # for k in params:
        #     if k not in constants:
        #         state['params'][k] = params[k]
        #     else:
        #         state['params'][k] = {
        #             'params': params[k],
        #             'constants': constants[k]
        #         }
        
        # for k in constants:
        #     if k not in params:
        #         state['constants'][k] = constants[k]
        
        # return state


    def get_apply_fns(self):
        return self._apply_fns

    def __getattribute__(self, name):
        if name in STATE_ORGANIZER_RESERVED:
            return super().__getattribute__(name)

        if name in self._apply_fns:
            apply_fns = self._apply_fns
            params = self._state['params']
            constants = self._state['constants']
            def apply(*args, **kwargs):
                x, next_state = apply_fns[name](
                    {
                        'params': params[name],
                        'constants': constants[name]
                    },
                    *args, **kwargs)

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
        if isinstance(value, tuple) and len(value) == 2:
            state, apply = value
            if not callable(apply):
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

        return super().__setattr__(name, value)        

# class StateOrganizer:

#     def __init__(self):
#         self._own_state = {
#             'params': {},
#             'constants': {},
#         }
#         self._t_own_state = {
#             'params': {},
#             'constants': {},
#         }

#         self._apply_fns = {}

#         self._t_params = {}
#         self._t_modules = {}

#         self._sub_modules = {}

#         self.config = SimpleNamespace()


#     def __setattr__(self, name, value):
#         if name == 'config' or name[0] == '_': # reserve names starting with _ to be assigned as normal.
#             return super().__setattr__(name, value)
#         # look, for now let's assume all attributes are also modules ok? thanks.

#         assert name not in self._own_state['params'], f"cannot create submodule {name}: a pre-existing parameter already has this name!"
#         assert name not in self._own_state['constants'], f"cannot create submodule {name}: a pre-existing constant already has this name!"

#         if len(value) == 4:
#             state, apply, t_params, t_module = value
#         else:
#             state, apply = value


#         self._sub_modules[name] = state
#         self._apply_fns[name] = apply

#         if len(value) == 4:
#             self._t_modules[name] = t_module
#             self._t_params[name] = t_params


#         return super().__setattr__(name, value)

#     def get_state(self):
#         params = {}
#         constants = {}

#         for name, value in self._sub_modules.items():
#             params[name] = value['params']
#             constants[name] = value['constants']

#         params.update(self._own_state['params'])
#         constants.update(self._own_state['constants'])

#         return {
#             'params': params,
#             'constants': constants
#         }

#     def get_t_state(self):
#         params = {}
#         constants = {}
#         for name, value in self._t_params.items():
#             params[name] = value['params']
#             constants[name] = value['constants']

#         params.update(self._t_own_state['params'])
#         constants.update(self._t_own_state['constants'])

#         return {
#             'params': params,
#             'constants': constants
#         }

#     def get_state_packer(self):
#         '''
#         returns a function that, given an appropriately shaped pytree object
#         will create a pytorch module-like object that binds  apply functions
#         to state data.


#         organizer.fc = Linear(10, 30)

#         # now, organizer.Linear is a tuple: organizer.Linear[0]['params'] = {'weight': weight, 'bias': bias}

#         f = organizer.get_state_packer()

#         # state is a pytree containing parameters corresponding to the
#         # parameters of oraganizer.fc: it has keys:
#         # state['params']['fc']['weight'] and
#         # state['params']['fc']['bias'] 
#         m = f(state)

#         # m is now a namespace such that  m.Linear is a *function* that
#         # applies a linear layer.

#         m.fc(input) # = output of linear layer with weight state['params']['fc']['weight'] and bias state['params']['fc']['bias'] 
#         '''
#         def state_packer(
#             apply_fns,
#             sub_module_names,
#             own_params_names,
#             own_constants_names,
#             config,
#             state
#             ):

#             sub_states = {
#                 k: {
#                     'params': state['params'][k],
#                     'constants': state['constants'][k]

#                 }
#                 for k in sub_module_names
#             }
#             values ={
#                     k: Partial(fn, sub_states[k])
#                 for k, fn in apply_fns.items()
#             }
#             values.update(
#                 {
#                     k: state['params'][k] for k in own_params_names
#                 }
#             )
#             values.update(
#                 {
#                     k: state['constants'][k] for k in own_constants_names
#                 }
#             )
#             values.update({'config': config})

#             packed_state = SimpleNamespace(**values)
        
#         return Partial(
#             state_packer,
#             self._apply_fns,
#             self._sub_modules.keys(),
#             self._own_state['params'].keys(),
#             self._own_state['constants'].keys(),
#             self.config,
#         )

#     def create_module(self, cls, t_module=None, return_torch=False):
#         state = self.get_state()
#         t_state = self.get_t_state()
#         apply_fn = Partial(cls.apply, self.get_state_packer())

#         if return_torch:
#             return state, apply_fn, t_state, t_module
#         else:
#             return state, apply_fn
#     # def get_all_t_params(self):
#     #     ret = {}
#     #     for name, value in self._t_modules.items():
#     #         ret[name] = value
#     #     ret.update(self._t_own_state['constants'])
#     #     ret.update(self._t_own_state['params'])
#     #     ret.update({'config': self.config})

#     #     return ret

#     def setup_t_module(self, t_module):
#         for name, value in self._t_modules.items():
#             t_module.__setattr__(name, value)
#         for name, value in self._t_own_state['params'].items():
#             t_module.register_parameter(name, torch.nn.Parameter(value))
#         for name, value in self._t_own_state['constants'].items():
#             t_module.register_buffer(name, value)

#         t_module.config = self.config
#         return t_module



#     def register_buffer(self, name, value, t_value=None):
#         assert name not in self._sub_modules, f"cannot register constant buffer {name}: a pre-existing submodule already has this name!"

#         self._own_state['constants'][name] = value

#         if t_value is not None:
#             self._t_own_state['constants'][name] = t_value

#     def register_parameter(self, name, value, t_value=None):
#         assert name not in self._sub_modules, f"cannot register parameter {name}: a pre-existing submodule already has this name!"

#         self._own_state['params'][name] = value

#         if t_value is not None:
#             self._t_own_state['params'][name] = t_value



# class TModule(torch.nn.Module):

#     def __init__(self, organizer):
#         super().__init__()
#         organizer.setup_t_module(self)


class Identity:
    def __new__(cls, rng=None):#, return_torch=None):
        # if return_torch is None:
        #     return_torch = RETURN_TORCH.value

        # t_module = torch.nn.Identity()
        # t_state = {
        #     'params': {},
        #     'constants': {},
        # }

        state = {
            'params': {},
            'constants': {}
        }


        # if return_torch:
        #     return state, cls.apply, t_state, t_module
        # else:
        #     del t_module
        #     del t_state
        return state, cls.apply

    def apply(state, x):
        return x, state

class Linear:
    
    def __new__(cls, in_features, out_features, bias=True, dtype=None, rng=None):#, return_torch=None):
        # if return_torch is None:
        #     return_torch = RETURN_TORCH.value

        # t_lin = torch.nn.Linear(in_features, out_features, bias)

        # t_params = {
        #     'weight': t_lin.weight
        # }
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
        return state, cls.apply


    def apply(state, input):
        params = state['params']

        weight = params['weight'].transpose()


        r = jnp.matmul(input, weight)

        if 'bias' in params:
            bias = params['bias']
            r = r + bias

        return r,  state




class Embedding:
    
    def __new__(cls, num_embeddings, embedding_dim, dtype=None, rng=None):#, return_torch=None):
        # if return_torch is None:
        #     return_torch = RETURN_TORCH.value

        # t_embed = torch.nn.Embedding(num_embeddings, embedding_dim)
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

        return state, cls.apply


    def apply(state, idx):
        weight = state['params']['weight']
        return weight[idx, :], state


class Sequential:
    '''
    chains together a list of state/apply_fn pairs ala torch.nn.Sequential
    
    arguments:
        states_and_applies: An iterable either of (state, apply_fn) tuples or 
            of (state, apply_fn, t_state, t_module), where each `state` is a pytree
            and each `apply_fn` is a function whose first argument is pytree of the 
            same shape as the corresponding `state`. If present, t_state is a pytree
            of pytorch tensors of the same shape (and likely same values) as `state`,
            and t_module is a pytorch module that implements the same function as apply_fn.

            If return_torch is True, then states_and_applies must contain t_state and t_module.

        return_torch: if True, return a pytorch Sequential module in addition to the
            Hax sequential information.
    returns:
        seq_state, apply_fn, and possibly also t_state, t_module.
        
    '''

    def __new__(cls, *states_and_applies, rng=None):#, return_torch=None):
        # if return_torch is None:
        #     return_torch = RETURN_TORCH.value

        if len(states_and_applies) == 0:
            raise ValueError(f"You must provide a non-empty list to Sequential!")
        


        states = [s_a[0] for s_a in states_and_applies]
        applies = [s_a[1] for s_a in states_and_applies]

        seq_state = group_state_list(states)
        apply_fn = functools.partial(cls.apply, applies)

        return seq_state, apply_fn


    def apply(applies, state, x):
        states = ungroup_state(state)

        next_states = []

        for s, f in zip(states, applies):
            x, state_update = f(s, x)
            next_states.append(state_update)

        return x, group_state_list(next_states)


class LayerNorm:
    
    def __new__(cls, normalized_shape, eps=1e-05, rng=None):#, return_torch=None):

        organizer = StateOrganizer()

        organizer.config.eps = 1e-05

        organizer.weight = jnp.ones(normalized_shape)

        organizer.bias = jnp.zeros(normalized_shape)

        return organizer.create_module(cls)




    def apply(pack_state, state, x):
        module = pack_state(state)

        e_x = jnp.average(x, axis=-1, keepdims=True)
        v_x = jnp.average((x-e_x)**2, axis=-1, keepdims=True)

        ln = (x - e_x)/jnp.sqrt(v_x + module.config.eps) * module.weight + module.bias

        return ln, module.get_state()


class Conv2d:
    def __new__(
        cls,
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
        # arguments = locals() # there's gotta be a better way...
        # arguments.pop('cls')
        # arguments.pop('rng')

        assert padding_mode=='zeros', "currently only the 'zeros' padding_mode is supported, sorry!"
        if rng is None:
            rng = rng_util.split()

        rng, subkey = jax.random.split(rng)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        state = {
            'params': {},
            'constants': {}
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

        conv_config = {
            'padding': padding,
            'stride': stride,
            'dilation': dilation,
            'feature_group_count': groups,
            'padding_mode': padding_mode
        }

        if bias:
            rng, subkey = jax.random.split(rng)
            state['params']['bias'] = jax.random.uniform(
                key=subkey,
                shape=(out_channels,),
                minval=-jnp.sqrt(k),
                maxval=jnp.sqrt(k),
            )

        return state, Partial(cls.apply, conv_config=conv_config)

    
    def apply(state, x, conv_config={
            'padding': 0,
            'stride': 1,
            'dilation': 1,
            'feature_group_count': 1,
            'padding_mode': 'same'
        }):
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

        constants = SimpleNamespace(**conv_config)

        


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


# class MultiheadAttention:

#     def __new__(
#         cls,
#         embed_dim,
#         num_heads,
#         bias=True,
#         add_bias_kv=False,
#         add_zero_attn=False,
#         kdim=None,
#         vdim=None,
#         batch_first=False,
#         device=None,
#         dtype=None,
#         return_torch=None):
#         '''
#         cls: class object
#         return_torch:  whether to return a pytorch object.
#         '''

#         arguments = locals() # there's gotta be a better way...
#         arguments.pop('cls')
#         arguments.pop('return_torch')
#         arguments.pop('rng')

#         if return_torch is None:
#             return_torch = RETURN_TORCH.value

#         organizer = StateOrganizer()



#         #might as well log the args...
#         organizer.config = arguments
#         t_mha = torch.nn.MultiheadAttention(**arguments)


#         # the pytorch implementation is full of random special cases.
#         # Let's try to not do that here. This requires one special case
#         # parameter extraction here, and then none later one.
#         if not t_mha._qkv_same_embed_dim
#             organizer.register_parameter(
#                 'q_proj_weight',
#                 t_to_jnp(t_mha.q_proj_weight)
#             )
#             organizer.register_parameter(
#                 'k_proj_weight',
#                 t_to_jnp(t_mha.k_proj_weight)
#             )
#             organizer.register_parameter(
#                 'v_proj_weight',
#                 t_to_jnp(t_mha.v_proj_weight)
#             )
#         else:
#             organizer.register_parameter(
#                 'q_proj_weight',
#                 t_to_jnp(t_mha.in_proj_weight[:embed_dim, :])
#             )
#             organizer.register_parameter(
#                 'k_proj_weight',
#                 t_to_jnp(t_mha.in_proj_weight[embed_dim:2*embed_dim, :])
#             )
#             organizer.register_parameter(
#                 'v_proj_weight',
#                 t_to_jnp(t_mha.in_proj_weight[2*embed_dim:, :])
#             )

#         if bias:
#             organizer.register_parameter(
#                 'in_proj_bias',
#                 t_to_jnp(t_mha.in_proj_bias)
#             )
        
#         if add_bias_kv:
#             organizer.register_parameter(
#                 'bias_k',
#                 t_to_jnp(t_mha.bias_k)
#             )
#             organizer.register_parameter(
#                 'bias_v',
#                 t_to_jnp(t_mha.bias_v)
#             )

#         constants = {
#             'dropout': dropout,
#             'num_heads': num_heads,
#             'add_zero_attn': add_zero_attn,
#             'batch_first': batch_first,
#         }
#         # if not self._qkv_same_embed_dim:
#         #     self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
#         #     self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
#         #     self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
#         #     self.register_parameter('in_proj_weight', None)
#         # else:
#         #     self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
#         #     self.register_parameter('q_proj_weight', None)
#         #     self.register_parameter('k_proj_weight', None)
#         #     self.register_parameter('v_proj_weight', None)

#         # if bias:
#         #     self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
#         # else:
#         #     self.register_parameter('in_proj_bias', None)
#         # self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

#         # if add_bias_kv:
#         #     self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         #     self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         # else:
#         #     self.bias_k = self.bias_v = None


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
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits -= jax.lax.stop_gradient(logits_max)

    no_ignore = jax.lax.stop_gradient(labels!=-100)

    ignore_labels = jnp.where(no_ignore, labels, jnp.zeros_like(labels))

    total = jax.lax.stop_gradient(jnp.sum(no_ignore))

    label_logits = jnp.take_along_axis(logits, ignore_labels[..., None], axis=-1)[..., 0]

    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    return jnp.sum(jnp.where(no_ignore, log_normalizers - label_logits, jnp.zeros_like(labels)))/total
