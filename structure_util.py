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
from jax.tree_util import Partial, tree_map

from jax._src.typing import Array, ArrayLike, DType, DTypeLike
from typing import overload, Any, Callable, Literal, Optional, Sequence, Tuple, Union

import rng_util


StructureTree = dict
PyTree = Any

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


def apply_tree(tree: StructureTree, global_config: dict, *args, **kwargs) -> [StructureTree, PyTree]:
    return tree['apply'](tree, global_config, *args, **kwargs)

# def apply(params: PyTree, buffers: PyTree, aux: dict, apply: dict, global_config: dict, *args, **kwargs) -> [StructureTree, PyTree]:
#     tree = merge_trees(params, buffers, aux, apply)
#     return apply_tree(tree, global_config, *args, **kwargs)

def bind_global_config(aux_and_apply, global_config: dict):
    organizer = StateOrganizer(aux_and_apply)
    def bound(params: PyTree, *args, **kwargs):
        merged = merge_trees(params, aux_and_apply)
        next_tree, output = apply_tree(merged, global_config, *args, **kwargs)
        next_params = filter_keys(next_tree)
        return next_params, output
    bound.aux_and_apply = aux_and_apply
    bound.bind_global_config = Partial(bind_global_config, aux_and_apply)
    return bound

def bind_module(tree: StructureTree, global_config: dict) -> [dict, Callable[[Any], Any]]:
    init_params, aux_and_apply = split_tree(tree, [RETURNED_KEYS,NON_RETURNED_KEYS])


    return init_params, bind_global_config(aux_and_apply, global_config)

def unbind_module(tree, bound):
    return merge_trees(tree, bound.aux_and_apply)

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


def split_tree(tree, key_sets=NON_CHILD_KEYS):
    key_sets = [[s] if isinstance(s, str) else s for s in key_sets]

    def split_func(node, path=None):
        return tuple({key: node[key] for key in s} for s in key_sets)

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
def is_jax_type(x):
    @jax.jit
    def jitted_id(a):
        return tree_map(lambda b: b, a)
    try:
        jitted_id(x)
    except:
        return False
    return True


def merge_configs(*configs):
    ret = {}
    for config in configs:
        ret.update(config)

    return ret

def create_tree_from_func(func):
    def wrapped_func(tree, global_config, *args, **kwargs):
        return tree, func(*args, **kwargs)

    return fill_tree({'apply': wrapped_func})


def organized_init(init_func):
        
    def decorated(*args, **kwargs):
        organizer = StateOrganizer()
        if 'rng' in kwargs:
            if kwargs['rng'] is None:
                kwargs['rng'] = rng_util.split()
            with rng_util.RNGState(kwargs['rng']):
                init_func(organizer, *args, **kwargs)
        else:
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
            submodule = StateOrganizer(state[CHILD_KEY][name], global_config)
            return submodule

        # check if name is unique:
        lookup = _inverse_lookup(state, name)
        assert len(lookup) <= 1
        if len(lookup) == 1:
            return state[lookup[0]][name]

        if name in REQUIRED_KEYS:
            return state[name]

        return super().__getattribute__(name)

    def __getitem__(self, name):
        return self.__getattribute__(name)
    
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
        elif is_jax_type(value):
            # by default we put things in params if they are jittable
            state['params'][name] = value
            return value


        return super().__setattr__(name, value)
