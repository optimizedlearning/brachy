'''
This file contains code for manipulating structure trees.
A structure tree is a dictionary containing keys:

params
buffers
aux
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

from jax._src.typing import Array, ArrayLike, DType, DTypeLike
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
    'aux',
    'apply',
]

NON_RETURNED_KEYS = [
    'aux',
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

# class HashableTree:
#     def __init__(self, tree):
#         self.tree = tree
#         # we save the current hash of the tree here, so that future in-place changes to the tree
#         # which might change the hash cannot fool us...
#         self.hash = self.hash_tree()
    
#     def hash_tree(self):
#         leaves, treedef = tree_flatten(self.tree)
#         leaves = tuple(leaves)
#         return hash((leaves, treedef))

#     def __hash__(self):
#         # add 1 to the hash so that nobody sneaky can construct the following tuple
#         # to engineer a collision (probably not a big deal, but anyway...)
#         return hash((self.hash, self.hash_tree())) + 1
    
#     def __eq__(self, other):
#         # technically, this equality could return False for trees that are
#         # really the same if they started out different and then were
#         # modified in place to become the same.
#         # However, a false negative will only result in an unecessary recompilation
#         # while a false positive will result in reusing an incorrect cached jitted function.

#         if not isinstance(other, HashableTree):
#             return False
#         eq = self.hash == other.hash
#         if not eq:
#             return False
#         try:
#             eq = tree_reduce(lambda a,b: a and b, tree_map(lambda x,y: x==y, self.tree, other.tree), True)
#         except:
#             return False
        
#         return eq

#     def __str__(self):
#         return f"HashableTree<{self.tree}>"


# # These are a bit hacky...
# def split_jax_nonjax(tree):
#     keep_static = lambda x: not valid_jaxtype(x) and isinstance(x, Hashable)
#     jaxtype_tree = tree_map(lambda x: None if keep_static(x) else x, tree)
#     nonjaxtype_tree = tree_map(lambda x: x if keep_static(x) else None, tree)

#     return jaxtype_tree, nonjaxtype_tree

# def merge_jax_nonjax(jax_tree, nonjax_tree):
#     def merge(jt, njt):
#         if njt == None:
#             return jt
#         else:
#             return njt

#     return tree_map(merge, jax_tree, nonjax_tree, is_leaf=lambda x: x is None)


# def improved_static(wrapper, *outer_args, static_argnums=None , static_argnames=None, **outer_kwargs):

#     wrapper_signature = inspect.signature(wrapper)
#     wrapper_parameters = wrapper_signature.parameters
#     wrapper_argnames = list(wrapper_parameters.keys())

#     if wrapper not in _CACHED_WRAPS:
#         _CACHED_WRAPS[wrapper] = {}

#     _CACHED_FUNCS = _CACHED_WRAPS[wrapper]

#     outer_static_argnums = static_argnums
#     outer_static_argnames = static_argnames



#     @wraps(wrapper)
#     def new_wrapper(fun, *wrapper_args, static_argnums=None , static_argnames=None, **wrapper_kwargs):

#         # Some checks to allow for default arguments specified in a decorator...
#         # this might be overly complicated a feature to have...
#         if outer_static_argnums is not None:
#             assert static_argnums is None, "ambiguous setting for static_argnums in wrapper {fun}!"
#             static_argnums = outer_static_argnums

#         if outer_static_argnames is not None:
#             assert static_argnames is None, "ambiguous setting for static_argnames in wrapper {fun}!"
#             static_argnames = outer_static_argnames

#         if len(outer_args) == 0:
#             assert len(wrapper_args) == 0, "ambiguous args for wrapper {fun}!"
#             wrapper_args = outer_args

#         for k,v in outer_kwargs.items():
#             assert k not in wrapper_kwargs, "ambiguous kwargs for wrapper {fun}!"
#             wrapper_kwargs[k] = v


#         if 'static_argnums' in wrapper_argnames and len(wrapper_args) >= wrapper_argnames.index('static_argnums'):
#             static_argnums = wrapper_args[wrapper_argnames.index('static_argnums')]
#         if 'static_argnames' in wrapper_argnames and len(wrapper_args) >= wrapper_argnames.index('static_argnames'):
#             static_argnames = wrapper_args[wrapper_argnames.index('static_argnames')]

#         if isinstance(static_argnums, int):
#             static_argnums = [static_argnums]
#         if isinstance(static_argnames, str):
#             static_argnames = [static_argnames]
        
#         if static_argnums is None:
#             static_argnums = []
#         if static_argnames is None:
#             static_argnames = []

#         static_argnums = list(static_argnums)

#         static_argnames = list(static_argnames)

#         signature = inspect.signature(fun)
#         parameters = signature.parameters
#         parameter_list = list(parameters.keys())

#         for name in static_argnames:
#             num = parameter_list.index(name)
#             if num not in static_argnums:
#                 if parameters[name].kind != parameters[name].KEYWORD_ONLY:
#                     static_argnums.append(num)
        
#         for num in static_argnums:
#             name = parameter_list[num]
#             if name not in static_argnames:
#                 if parameters[name].kind != parameters[name].POSITIONAL_ONLY:
#                     static_argnames.append(name) 

#         if fun not in _CACHED_FUNCS:
#             _CACHED_FUNCS[fun] = {}

#         cached_calls = _CACHED_FUNCS[fun]
        
#         @wraps(fun)
#         def wrapped_fun(*args, **kwargs):
            
#             split_args = []
#             structure_tree_args_statics = {}
#             structure_tree_kwargs_statics = {}

#             tree_args_statics = {}
#             tree_kwargs_statics = {}
#             for argnum, arg in enumerate(args):
#                 if not is_structure_tree(arg):

#                     jax_tree, nonjax_tree = split_jax_nonjax(arg)
#                     split_args.append(jax_tree)
#                     tree_args_statics[argnum] = nonjax_tree
#                 else:
#                     params_buffers, rest = split_tree(arg, [RETURNED_KEYS, NON_RETURNED_KEYS])
#                     split_args.append(params_buffers)
#                     structure_tree_args_statics[argnum] = rest        

#             split_kwargs = {}
#             for k, v in kwargs.items():
#                 if not is_structure_tree(v):
#                     jax_tree, nonjax_tree = split_jax_nonjax(v)
#                     split_kwargs[k] = jax_tree
#                     tree_kwargs_statics[k] = nonjax_tree
#                 else:
#                     params_buffers, rest = split_tree(v, [RETURNED_KEYS, NON_RETURNED_KEYS])
#                     split_kwargs[k] = params_buffers
#                     structure_tree_kwargs_statics[k] = rest  


#             static_args = [arg if i in static_argnums else None for i, arg in enumerate(split_args)]
#             static_kwargs = {
#                 k: split_kwargs.get(k) for k in static_argnames
#             }

#             cache_key = HashableTree({
#                 'static_argnums': static_argnums,
#                 'static_args': static_args,
#                 'static_argnames': static_argnames,
#                 'static_kwargs': static_kwargs,
#                 'structure_tree_args_statics': structure_tree_args_statics,
#                 'tree_statics': tree_args_statics,
#                 'structure_tree_kwargs_statics': structure_tree_kwargs_statics,
#                 'tree_kwargs_statics': tree_kwargs_statics,
#             })


#             if cache_key not in cached_calls:
#                 def to_wrap(*args, **kwargs):
#                     args_with_statics = list(args)
#                     for i in range(len(args)):
#                         if i in static_argnums:
#                             args_with_statics[i] = static_args[i]
#                         if i in structure_tree_args_statics:
#                             args_with_statics[i] = merge_trees(args_with_statics[i], structure_tree_args_statics[i])
#                         if i in tree_args_statics:
#                             args_with_statics[i] = merge_jax_nonjax(args_with_statics[i], tree_args_statics[i])
                    
#                     kwargs_with_statics = dict(kwargs)
#                     for k in kwargs:
#                         if k in static_argnames:
#                             kwargs_with_statics[k] = static_kwargs[k]
#                         if k in structure_tree_kwargs_statics:
#                             kwargs_with_statics[k] = merge_trees(kwargs_with_statics[k], structure_tree_args_statics[k])
#                         if k in tree_kwargs_statics:
#                             kwargs_with_statics[k] = merge_jax_nonjax(kwargs_with_statics[k], tree_kwargs_statics[k])
#                     values = fun(*args_with_statics, **kwargs_with_statics)

#                     if not isinstance(values, tuple):
#                         values = [values]
                    
#                     # this should happen upon first tracing to populate the static parts of any structure trees
#                     # in the returned values.
#                     returned_structure_statics = {}
#                     split_values = list(values)
#                     for i, v in enumerate(values):
#                         if is_structure_tree(v):
#                             jax_tree, nonjax_tree = split_jax_nonjax(v)
#                             if cached_calls[cache_key]['returned_structure_statics'] is None:
#                                 returned_structure_statics[i] = nonjax_tree
#                             split_values[i] = jax_tree
#                     if cached_calls[cache_key]['returned_structure_statics'] is None:
#                         cached_calls[cache_key]['returned_structure_statics'] = returned_structure_statics
#                     if len(split_values) == 1:
#                         return split_values[0]
#                     else:
#                         return tuple(split_values)

#                 cached_calls[cache_key] = {}
#                 cached_calls[cache_key]['wrapped_func'] = wrapper(to_wrap, *wrapper_args, **wrapper_kwargs)
#                 cached_calls[cache_key]['returned_structure_statics'] = None

#             wrapped = cached_calls[cache_key]['wrapped_func']

#             args_without_statics  = list(split_args)
#             for i in static_argnums:
#                 if i < len(args):
#                     args_without_statics[i] = None
#             kwargs_without_statics = dict(split_kwargs)
#             for k in static_argnames:
#                 if k in kwargs:
#                     kwargs_without_statics[k] = None
            
#             values = wrapped(*args_without_statics, **kwargs_without_statics)

#             if not isinstance(values, tuple):
#                 values = [values]
#             values = list(values)
            
#             for i, v in cached_calls[cache_key]['returned_structure_statics'].items():
#                 values[i] = merge_jax_nonjax(values[i], v)
            
#             if len(values) == 1:
#                 return values[0]
#             else:
#                 return tuple(values)


#         return wrapped_fun

#     return new_wrapper


# jit = improved_static(jax.jit)

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
    split_fn = lambda x: split_tree(x, ['params', ['buffers', 'aux', 'apply']]),
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
            output_type = rest['aux']['mixed_precision']['output_type']
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



def bind_global_config(aux_and_apply, global_config: dict):
    organizer = StateOrganizer(aux_and_apply)
    def bound(params: PyTree, *args, **kwargs):
        merged = merge_trees(params, aux_and_apply)
        next_tree, output = apply_tree(merged, global_config, *args, **kwargs)
        next_params = filter_keys(next_tree)
        return next_params, output
    bound.aux_and_apply = aux_and_apply
    bound.global_config = global_config
    bound.bind_global_config = Partial(bind_global_config, aux_and_apply)
    return bound

def bind_module(tree: StructureTree, global_config: dict) -> [dict, Callable[[Any], Any]]:
    init_params, aux_and_apply = split_tree(tree, [RETURNED_KEYS,NON_RETURNED_KEYS])
    init_params = tree_map(lambda x: jnp.array(x), init_params)
    init_params = tree_map(lambda x: jnp.array(x, dtype=x.dtype), init_params)


    return init_params, bind_global_config(aux_and_apply, global_config)

def unbind_module(state, bound):
    return merge_trees(state, bound.aux_and_apply), bound.global_config

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
    other_keys.remove('aux')
    other_keys.remove('apply')

    return split_tree(tree, key_sets=['params', other_keys, ['aux', 'apply']])


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

    def recursive_register_aux(self, name, value):
        self.register_aux(name, value)
        for m in self.submodules().values():
            m.recursive_register_aux(name, value)
    
    def __call__(self, *args, **kwargs):
        state = self._state
        global_config = self.get_global_config()
        next_state, output = state['apply'](state, global_config, *args, **kwargs)

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

    def register_aux(self, name, value):
        self._state['aux'][name] = value

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
