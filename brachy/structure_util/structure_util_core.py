'''
This file contains code for manipulating structure trees.
A structure tree is a dictionary containing keys:

params
buffers
static
apply
submodules

With the exception of apply, the value associated with each key is itself a dict.
The value associated with apply should be a function with signature:
def apply(structure_tree, global_config, ...) -> structure_tree, Any



'''


import jax
from jax import numpy as jnp
from jax.tree_util import Partial, tree_map, tree_flatten, tree_reduce
from jax.core import valid_jaxtype

from jax import Array
import typing
from typing import (Any, Callable, Generator, Hashable, Iterable, List, Literal,
                    NamedTuple, Optional, Sequence, Tuple, TypeVar, Union,
                    overload, cast)
AxisName = Hashable

from brachy import rng_util
from json import dumps, loads

from functools import wraps

import inspect


StructureTree = dict
PyTree = Any

_CACHED_WRAPS = {}

CHILD_KEY = 'submodules'

STATE_ORGANIZER_RESERVED = [
    '_state',
    '_global_config',
    '_submodule_global_configs',
]


NON_CHILD_KEYS = [
    'params',
    'buffers',
    'static',
    'apply',
]

NON_RETURNED_KEYS = [
    'static',
    'apply'
]

RETURNED_KEYS = [
    k for k in NON_CHILD_KEYS if k not in NON_RETURNED_KEYS
]

REQUIRED_KEYS = NON_CHILD_KEYS + [CHILD_KEY]


def value_and_grad(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
        outnum: int = 0, has_aux: bool = False, holomorphic: bool = False,
        allow_int: bool = False,
        reduce_axes: Sequence[AxisName] = ()) -> Callable:
    if outnum == 0:
        return jax.value_and_grad(fun, argnums, has_aux, holomorphic, allow_int, reduce_axes)
    
    def reorded_fun(*args, **kwargs):
        output = fun(*args, **kwargs)

        output_to_diff = output[outnum]

        other_indices = list(range(len(output)))
        other_indices.pop(output_num)
    
        other_outputs = [output[i] for i in extra_indices]
        reordeded_output = output[outnum], tuple(other_outputs)

    raw_value_and_grad_fn = jax.value_and_grad(reordered_fun, argnums, True, holomorphic, allow_int, reduce_axes)


    def value_and_grad_fn(*args, **kwargs):
        (output, aux_output), grad = raw_value_and_grad_fn(*args, **kwargs)


        final_output = tuple([aux for aux in aux_output[:output_num]] + [output] + [aux for aux in aux_output[output_num:]])

        return final_output, grad

def map_params_buffers(f, tree):
    def node_f(node, path=None):
        node = dict(node)
        node['params'] = tree_map(f, node['params'])
        node['buffers'] = tree_map(f, node['buffers'])
        node['submodules'] = {}
        return node
    return structure_tree_map(node_f, tree)

def uncast_mixed_precision(state, grad):
    # types = state['buffers']['mixed_precision']['types']
    loss_scalar = state['buffers']['mixed_precision']['loss_scalar']

    def uncast_and_scale(g, path):# t, path):
        g = copy_to_leaf(g)
        # t = copy_to_leaf(t)
        g['params'] = tree_map(lambda x: x / loss_scalar.astype(x.dtype), g['params'])
        return g

    return structure_tree_map(uncast_and_scale, grad)#, types)

# TODO: allow a list of argnums
def tree_value_and_grad(
    fun: Callable,
    output_num: int=0,
    argnums: int=0,
    split_fn = lambda x: split_tree(x, ['params', ['buffers', 'static', 'apply']]),
    merge_fn = lambda x, y: merge_trees(x, y)
    ):
    def fun_to_differentiate(*args, **kwargs):
        args = list(args)
        args[argnums] = merge_fn(args[argnums], kwargs['_structure_tree_nondiff_'])
        del kwargs['_structure_tree_nondiff_']

        
        tree, *output = fun(*args, **kwargs)
        output_to_diff = output[output_num]

        if 'mixed_precision' in tree['buffers']:
            output_to_diff = output_to_diff * tree['buffers']['mixed_precision']['loss_scalar']

        extra_indices = list(range(len(output)))
        extra_indices.pop(output_num)
        output = [tree] + [output[i] for i in extra_indices]
        return output_to_diff,  tuple(output)
       

    prelim_value_and_grad_fn = jax.value_and_grad(fun_to_differentiate, argnums=argnums, has_aux=True)

    def value_and_grad_fn(*args, **kwargs):
        args = list(args)
        tree, rest = split_fn(args[argnums])
        kwargs['_structure_tree_nondiff_'] = rest
        args[argnums] = tree
        (output, (new_tree, *aux_output)), grad = prelim_value_and_grad_fn(*args, **kwargs)
        if 'mixed_precision' in new_tree['buffers']:
            mixed_precision = new_tree['buffers']['mixed_precision']
            grad = uncast_mixed_precision(new_tree, grad)
            output_type = rest['static']['mixed_precision']['output_type']
            output = output.astype(output_type) / mixed_precision['loss_scalar'].astype(output_type)  

        final_output = tuple([aux for aux in aux_output[:output_num]] + [output] + [aux for aux in aux_output[output_num:]])

        return (new_tree, *final_output), grad

    return value_and_grad_fn 

def state_value_and_grad(fun, output_num=0):
    
    def processed_grad_fn(state, *args, **kwargs):
        params, buffers = split_tree(state, ['params', 'buffers'])

        def fun_to_differentiate(params):
            state = merge_trees(params, buffers)
            state, *output = fun(state, *args, **kwargs)
            output_to_diff = output[output_num]

            if 'mixed_precision' in state['buffers']:
                output_to_diff = output_to_diff * state['buffers']['mixed_precision']['loss_scalar']

            extra_indices = list(range(len(output)))
            extra_indices.pop(output_num)
            output = [state] + [output[i] for i in extra_indices]
            return output_to_diff,  tuple(output)

        grad_fn = jax.value_and_grad(fun_to_differentiate, has_aux=True)

        (output, (new_state, *aux_output)), grad = grad_fn(params)

        if 'mixed_precision' in new_state['buffers']:
            mixed_precision = new_state['buffers']['mixed_precision']
            grad = uncast_mixed_precision(new_state, grad)
            output = output.astype(mixed_precision['output_type'].dtype) / mixed_precision['loss_scalar'].astype(mixed_precision['output_type'].dtype)


        final_output = tuple([aux for aux in aux_output[:output_num]] + [output] + [aux for aux in aux_output[output_num:]])

        return (new_state, *final_output), grad

    return processed_grad_fn


def apply_tree(tree: StructureTree, global_config: dict, *args, **kwargs) -> [StructureTree, PyTree]:
    return tree['apply'](tree, global_config, *args, **kwargs)

def apply_and_update_tree(tree: StructureTree, global_config: dict, *args, **kwargs) -> [StructureTree, PyTree]:
    next_tree, *x = tree['apply'](tree, global_config, *args, **kwargs)
    next_tree = update_tree(tree, next_tree)
    return next_tree, *x



def bind_global_config(static_and_apply, global_config: dict):
    organizer = StateOrganizer(static_and_apply)
    def bound(params: PyTree, *args, **kwargs):
        merged = merge_trees(params, static_and_apply)
        next_tree, output = apply_tree(merged, global_config, *args, **kwargs)
        next_params = filter_keys(next_tree)
        return next_params, output
    bound.static_and_apply = static_and_apply
    bound.global_config = global_config
    bound.bind_global_config = Partial(bind_global_config, static_and_apply)
    return bound

def bind_module(tree: StructureTree, global_config: dict) -> [dict, Callable[[Any], Any]]:
    init_params, static_and_apply = split_tree(tree, [RETURNED_KEYS,NON_RETURNED_KEYS])
    init_params = tree_map(lambda x: jnp.array(x), init_params)
    init_params = tree_map(lambda x: jnp.array(x, dtype=x.dtype), init_params)


    return init_params, bind_global_config(static_and_apply, global_config)

def unbind_module(state, bound):
    return merge_trees(state, bound.static_and_apply), bound.global_config

def is_structure_tree(tree, recurse=False):
    if not isinstance(tree, dict):
        return False
    if set(tree.keys()) != set(REQUIRED_KEYS):
        return False
    for key in REQUIRED_KEYS:
        if key not in tree:
            return False
        if key == 'apply':
            if not callable(tree[key]):
                return False
        elif not isinstance(tree[key], dict):
            return False


    if is_leaf(tree):
        return True
    
    if recurse:
        for k in tree[CHILD_KEY]:
            if not is_structure_tree(tree[CHILD_KEY][k], recurse=True):
                return False
                

    return True

def copy_to_leaf(node):
    leaf = {k: v for k,v in node.items()}
    leaf[CHILD_KEY] = {}
    return leaf

def children(tree):
    return tree[CHILD_KEY]

def is_leaf(tree):
    return tree[CHILD_KEY] == {} # not in tree or tree[CHILD_KEY] in [{}, []] # just stop supporting this nonsense...

# probably there is a "correct" way to do this, but I don't know the syntax 
def tupleize(x):
    if isinstance(x, tuple):
        return x
    else:
        return (x,)

def structure_tree_map(func, *trees, path=None):
    if path is None:
        path = []
    # mapped_tree = {}
    mapped_tree = func(*trees, path=path)

    
    # Tricky: we need to overwrite mapped_tree[CHILD_KEY] even if it is already {} since
    # there might be some pointer snafus otherwise.
    # mapped_tree = tu
    if isinstance(mapped_tree, tuple):
        for m in mapped_tree:
            assert CHILD_KEY not in m or is_leaf(m), "tree_map func must return leaf nodes!"
            m[CHILD_KEY] = {}
    else:
        assert CHILD_KEY not in mapped_tree or is_leaf(mapped_tree), "tree_map func must return leaf nodes!"
        mapped_tree[CHILD_KEY] = {}

    all_children = {}
    for tree in trees:
        for key, child in tree[CHILD_KEY].items():
            if key not in all_children:
                all_children[key] = []
            all_children[key].append(child)

    for key, children in all_children.items():
        mapped_child = structure_tree_map(func, *children, path=path+[key])
        if isinstance(mapped_tree, tuple):
            for i in range(len(mapped_tree)):
                mapped_tree[i][CHILD_KEY][key] = mapped_child[i]
        else:
            mapped_tree[CHILD_KEY][key] = mapped_child
        
    return mapped_tree

def filter_keys(tree, *keys):
    if len(keys) == 0:
        keys = ['params', 'buffers']


    def filter_func(node, path):
        return {
            key: node[key] for key in keys
        }
    return structure_tree_map(filter_func, tree)



def get_children(tree):
    return tree[CHILD_KEY]

def copy_dict(d):
    return {k: v for k, v in d.items()}


def fill_tree(tree):
    '''
    fills missing fields in a tree with default empty values.
    Returns a new tree (does not modify the old one in place).
    '''
    filled_tree = copy_dict(tree)
    empty = empty_tree()
    for key in REQUIRED_KEYS:
        if key not in filled_tree:
            filled_tree[key] = empty[key]
    return filled_tree

def empty_tree(tree=None):
    empty = {key: {} for key in REQUIRED_KEYS}
    empty['apply'] = lambda t, g, x: (t, x)
    if tree is None:
        return empty

    return structure_tree_map(lambda t, path=None: {k:v for k, v in empty.items()}, tree)

@Partial
def update_tree(tree, next_tree):
    return merge_trees(tree, next_tree, keys_to_override=RETURNED_KEYS)

@Partial
def get_tree_update(tree):
    return filter_keys(tree, *RETURNED_KEYS)

@Partial
def merge_trees(*trees, keys_to_merge=NON_CHILD_KEYS, keys_to_override=NON_CHILD_KEYS):

    if len(trees) == 0:
        return merged

    def merge_func(*trees, path=None):
        merged  = {}
        for tree in trees:
            for key in tree:
                if key == CHILD_KEY:
                    continue
                if key not in keys_to_merge:
                    continue
                if key not in keys_to_override and key in merged:
                    continue
                if key == 'apply':
                    merged[key] = tree[key]
                    continue
                if key not in merged:
                    merged[key] = {}
                merged[key].update(tree[key])
        return merged

    return structure_tree_map(merge_func, *trees)

@Partial
def get_params(tree):
    def split_func(node, path):
        return {'params': node['params']}, {key: node[key] for key in node if key not in ['params', CHILD_KEY]}
    return structure_tree_map(split_func, tree)

def split_non_static(tree):
    return split_tree(tree, [RETURNED_KEYS, NON_RETURNED_KEYS])

def split_tree(tree, key_sets=NON_CHILD_KEYS):
    if isinstance(key_sets, str):
        return filter_keys(tree, key_sets)
    else:
        key_sets = [[s] if isinstance(s, str) else s for s in key_sets]

    def get_keys(node, s):
        if s is None:
            return {key: node[key] for key in node if key != CHILD_KEY}
        else:
            return {key: node[key] for key in s}

    def split_func(node, path=None):
        return tuple(get_keys(node, s) for s in key_sets)

    return structure_tree_map(split_func, tree)

    # return [filter_keys(tree, *s) for s in key_sets]

def split_params(tree):
    other_keys = [_ for  _ in NON_CHILD_KEYS]
    other_keys.remove('params')
    other_keys.remove('static')
    other_keys.remove('apply')

    return split_tree(tree, key_sets=['params', other_keys, ['static', 'apply']])


def _inverse_lookup(tree, name):
    lookup = []
    for key in tree:
        if key == 'apply':
            continue
        if name in tree[key]:
            lookup.append(key)
    return lookup


def _is_valid_submodule(v):
    return isinstance(v, tuple) and  len(v) == 2  and is_structure_tree(v[0]) and isinstance(v[1], dict)


#THIS FEELS SUPER HACKY
def is_jax_tree(x):
    return tree_reduce(lambda a,b: a and valid_jaxtype(b), x, True)
    # @jax.jit
    # def jitted_id(a):
    #     return tree_map(lambda b: b, a)
    # try:
    #     jitted_id(x)
    # except:
    #     return False
    # return True


def merge_configs(*configs):
    ret = {}
    for config in configs:
        ret.update(config)

    return ret

def create_tree_from_func(func):
    def wrapped_func(tree, global_config, *args, **kwargs):
        return tree, func(*args, **kwargs)

    return fill_tree({'apply': wrapped_func})




def fill_tree_from_torch_module(tree, torch_module, get_grad=False):
    def t_to_np(tensor):
        try:
            return tensor.detach().numpy()
        except:
            return tensor

    def extract_params(node, path):
        module = torch_module
        for name in path:
            if isinstance(name, int):
                module = module[name]
            else:
                module = getattr(module, name)
        new_node = {
            key: {subkey: value for subkey, value in node[key].items()} for key in node if key != CHILD_KEY and isinstance(node[key], dict)
        }
        if 'apply' in node:
            new_node['apply'] =  node['apply']
        param_process = lambda x: x
        if get_grad:
            param_process = lambda x: x.grad

        if 'params' in node:
            for p in node['params']:
                new_node['params'][p] = t_to_np(param_process(getattr(module, p)))
        if 'buffers' in node:
            for p in node['buffers']:
                if hasattr(module, p):
                    new_node['buffers'][p] = t_to_np(getattr(module, p))
        return new_node

    return structure_tree_map(extract_params, tree)



def organized_init(init_func):
    def decorated(*args, **kwargs):
        organizer = StateOrganizer()
        init_func(organizer, *args, **kwargs)
        return organizer.create_module()
    return decorated

def organized_init_with_rng(init_func):
    signature = inspect.signature(init_func)
    parameters = signature.parameters
    takes_rng = 'rng' in parameters
    rng_index  = list(parameters.keys()).index('rng')
    def decorated(*args, **kwargs):
        organizer = StateOrganizer()
        if not takes_rng:
            init_func(organizer, *args, **kwargs)
            return organizer.create_module()
        if len(args) >= rng_index:
            if args[rng_index] is None:
                rng = rng_util.split()
                args[rng_index] = rng
            else:
                rng = args[rng_index]
        else:
            if kwargs.get('rng') is None:
                rng = rng_util.split()
                kwargs['rng'] = rng
            else:
                rng = kwargs['rng']
        with rng_util.RNGState(rng):
            init_func(organizer, *args, **kwargs)
        return organizer.create_module()
    return decorated
        

def organized_apply(apply_func):
    def decorated(tree, global_config, *args, **kwargs):
        
        organizer = StateOrganizer(tree, global_config)
        output = apply_func(organizer, *args, **kwargs)

        return organizer.get_state(), output
    
    return decorated
        

class StateOrganizer:

    def __init__(
        self,
        state=None,
        global_config=None
        ):
        if state is None:
            state = {
                key: {} for key in REQUIRED_KEYS
            }
        if global_config is None:
            global_config = {}

        self._state = state
        self._global_config = global_config
        self._submodule_global_configs = {}
        
    def update_global_config(self, update, *args):
        if len(args) == 0:
            self._global_config.update(update)
        elif len(args) == 1:
            self._global_config[update] = args[0]
        else:
            raise SyntaxError("too many arguments to update_global_config!")

    def create_module(self, apply=None):
        if apply is not None:
            self._state['apply'] = apply
        return self.get_state(), self.get_global_config()

    def set_apply(self, apply=None):
        if apply is not None:
            self._state['apply'] = apply
    
    def set_forward(self, apply=None):
        return self.set_apply(apply)

    def get_state(self):
        return self._state
    
    def get_tree(self):
        return self._state

    def get_state_update(self):
        return filter_keys(self._state)

    def set_train_mode(self, mode):
        self.update_global_config('train_mode', mode)

    def get_global_config(self, key=None):
        
        global_config = {}
        for submodule, config in self._submodule_global_configs.items():
            global_config.update(config)
        global_config.update({k: v for k, v in self._global_config.items()})
        if key is None:
            return global_config
        
        return global_config[key]

    def get_apply_fns(self):
        return self._apply_fns

    def __getattribute__(self, name):
        if name in STATE_ORGANIZER_RESERVED:
            return super().__getattribute__(name)

        # we've already dealt with self._state and self._global_config, so now
        # it's safe to access them.
        state = self._state

        global_config = self._global_config

        if name in state[CHILD_KEY]:
            submodule = StateOrganizer(state[CHILD_KEY][name], self.get_global_config())
            return submodule

        # check if name is unique:
        lookup = _inverse_lookup(state, name)
        assert len(lookup) <= 1
        if len(lookup) == 1:
            return state[lookup[0]][name]

        if name in REQUIRED_KEYS and name != CHILD_KEY:
            return state[name]

        return super().__getattribute__(name)

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def submodules(self):
        state = self._state
        return {
            k: StateOrganizer(state[CHILD_KEY][k], self.get_global_config()) for k in state[CHILD_KEY]
        }

    def recursive_register_static(self, name, value):
        self.register_static(name, value)
        for m in self.submodules().values():
            m.recursive_register_static(name, value)
    
    def __call__(self, *args, **kwargs):
        state = self._state
        global_config = self.get_global_config()
        values = state['apply'](state, global_config, *args, **kwargs)
        next_state = values[0]
        if len(values) == 1:
            output = None
        if len(values) == 2:
            output = values[1]
        else:
            output = values[1:]

        # Tricky: we must change the keys of self._state directly: we cannot simply reassign state
        # as self._state = merge(self._state, next_state, keys_to_override=['params','buffers'])
        # because self._state may be pointed to by a parent StateOrganizer and we need these state
        # changes to be propogated up to the parent's ._state
        self._state['params'] = next_state['params']
        self._state['buffers'] = next_state['buffers']


        return output
    
    def update_tree(self, update):
        self._state.update(merge_trees(self._state, update))

    def register_parameter(self, name, value):
        self._state['params'][name] = value

    def register_buffer(self, name, value):
        self._state['buffers'][name] = value

    def register_static(self, name, value):
        self._state['static'][name] = value

    def register_submodule(self, name, value):
        assert _is_valid_submodule(value)
        self._state[CHILD_KEY][name] = value[0]
        self._submodule_global_configs[name] = value[1]

    def __setattr__(self, name, value):
        '''
        sets an attribute.
        We assume that value is EITHER a:
        1. tuple (tree, global_config) corresponding to the initial structure  tree
            and global_config of another module.
        2. a pytree.

        in either case, the state info is stored as a trainable parameter.
        To make a non-trainable parameter, you must use register_buffer, as in pytorch.å
        '''
        if name in STATE_ORGANIZER_RESERVED:
            return super().__setattr__(name, value)

        state = self._state
        lookup = _inverse_lookup(self._state, name)

        if len(lookup) > 1:
            raise ValueError("attempting to set a value with an ambiguous name!")
        if len(lookup) == 1:
            lookup = lookup[0]
            if lookup == CHILD_KEY:
                self.register_submodule(name, value)
            else:
                state[lookup][name] = value
            return value

        # this name does not appear yet.
        if _is_valid_submodule(value):
            self.register_submodule(name, value)
            return value
        elif callable(value):
            # if you try to make a function attribute, we will create a new submodule
            # for it.
            self.register_submodule(name, create_tree_from_func(value))
        elif is_jax_tree(value):
            # by default we put things in params if they are jittable
            state['params'][name] = value
            return value

        # if name == 'betas':
        #     print("found nothing..")
        return super().__setattr__(name, value)
