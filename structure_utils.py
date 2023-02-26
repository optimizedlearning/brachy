
import jax
from jax.tree_util import Partial, tree_map

from jax._src.typing import Array, ArrayLike, DType, DTypeLike
from typing import overload, Any, Callable, Literal, Optional, Sequence, Tuple, Union

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
    'aux',
    'constants',
    'apply',
]

NON_RETURNED_KEYS = [
    'constants',
    'apply'
]

RETURNED_KEYS = [
    k for k in NON_CHILD_KEYS if k not in NON_RETURNED_KEYS
]

REQUIRED_KEYS = NON_CHILD_KEYS + [CHILD_KEY]


def apply_tree(tree: StructureTree, global_config: dict, *args, **kwargs) -> [StructureTree, PyTree]:
    return tree['apply'](tree, global_config, *args, **kwargs)

def apply(params: PyTree, aux: PyTree, constants: dict, apply: dict, global_config: dict, *args, **kwargs) -> [StructureTree, PyTree]:
    tree = merge_trees(params, aux, constants, apply)
    return apply_tree(tree, global_config, *args, **kwargs)

def bind_module(tree: StructureTree, global_config: dict):
    init_params, consts_and_apply = split_tree(tree, [RETURNED_KEYS,NON_RETURNED_KEYS])


    def bound(params: PyTree, *args, **kwargs) -> [PyTree, PyTree, PyTree]:
        merged = merge_trees(params, consts_and_apply)
        next_tree, output = apply_tree(merged, global_config, *args, **kwargs)
        next_params = filter_keys(next_tree)
        return next_params, output

    bound.consts_and_apply = consts_and_apply

    return init_params, bound

def unbind_module(tree, bound):
    return merge_trees(tree, bound.consts_and_apply)

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
    

def structure_tree_map(tree, func, initial_path=[]):
    mapped_tree = {}
    mapped_tree = func(tree, initial_path)

    if CHILD_KEY not in mapped_tree:
        mapped_tree[CHILD_KEY] = {}

    assert is_structure_tree(mapped_tree), "mapping function in tree_map did not return a valid structure tree!"

    if not is_leaf(tree):
        assert is_leaf(mapped_tree), "tree_map func must return leaf nodes!"
        children = tree[CHILD_KEY]
        mapped_tree[CHILD_KEY] = {key: tree_map(child, initial_path+[key]) for key, child in children.items()}
        
    return mapped_tree

def filter_keys(tree, *keys):
    if len(keys) == 0:
        keys = ['params', 'aux']
    base = {
        key: tree[key] for key in keys
        }
    children = tree[CHILD_KEY]

    base[CHILD_KEY] = {
        name: filter_keys(child, *keys) for name, child in children.items()
        }

    return base


def get_children(tree):
    return tree[CHILD_KEY]

def empty_like(tree, key):
    empty_tree = {
        key: {},
        CHILD_KEY: {}
    }
    if not is_leaf(tree):
        empty_tree[CHILD_KEY] = empty_like(tree[CHILD_KEY], key)
    return empty_tree

def merge_trees(*trees, keys_to_merge=NON_CHILD_KEYS):
    merged = {}

    if len(trees) == 0:
        return merged

    non_leaves  = []
    for tree in trees:
        if not is_leaf(tree):
            non_leaves.append(tree[CHILD_KEY])
        for key in tree:
            if key == CHILD_KEY or key not in keys_to_merge:
                continue
            if key == 'apply':
                merged[key] = tree[key]
                continue
            if key not in merged:
                merged[key] = {}
            merged[key].update(tree[key])

    if len(non_leaves) != 0:
        children_names = set(non_leaves[0].keys())
        for nl in non_leaves[1:]:
            assert children_names == set(nl.keys()), "trees to merge have differing structures!"


        merged[CHILD_KEY] = {
            k: merge_trees(*[n[k] for n in non_leaves], keys_to_merge=keys_to_merge) for k in children_names
        }
    else:
        merged[CHILD_KEY] = {}

    return merged


def split_tree(tree, key_sets=NON_CHILD_KEYS):
    key_sets = [[s] if isinstance(s, str) else s for s in key_sets]
    return [filter_keys(tree, *s) for s in key_sets]

def split_params(tree):
    other_keys = [_ for  _ in NON_CHILD_KEYS]
    other_keys.remove('params')
    other_keys.remove('constants')
    other_keys.remove('apply')

    return split_tree(tree, key_sets=['params', other_keys, ['constants', 'apply']])

class DotDict:
    def __init__(
        self,
        **kwargs
    ):
        self.__dotdict_data = kwargs


    def __getattribute__(self, name):
        if name != '__dotdict_data':
            data = self.__dotdict_data
            if name in data:
                return data
        
        return super().__getattribute__(name)

    def __getitem__(self, key):
        return self.__dotdict_data[key]

    
    def __setattr__(self, name, value):
        self.__dotdict_data[name] = value



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

    def get_state(self):
        return self._state

    def get_state_update(self):
        return filter_keys(self._state)

    def get_global_config(self, key=None):
        global_config = {}
        for submodule, config in self._submodule_global_configs.items():
            global_config.update(config)
        global_config.update({k: v for k, v in self._global_config.items()})
        if key is None:
            return global_config
        
        return global_config[key]

    # def create_module(self):
    #     return self._state, self.get_global_config()

    def get_apply_fns(self):
        return self._apply_fns

    def __getattribute__(self, name):
        if name in STATE_ORGANIZER_RESERVED:
            return super().__getattribute__(name)

        # we've already dealt with self._state and self._global_config, so now
        # it's safe to access it them.
        state = self._state

        global_config = self._global_config

        if name in state[CHILD_KEY]:
            submodule = StateOrganizer(state[CHILD_KEY][name], global_config)
            def apply(*args, **kwargs):
                output = submodule(*args, **kwargs)
                state[CHILD_KEY][name] = merge_trees(state[CHILD_KEY][name], submodule.get_state(), keys_to_merge=['params', 'aux'])
                return output
            return apply

        # check if name is unique:
        lookup = _inverse_lookup(state, name)
        if len(lookup) == 1:
            return state[lookup[0]][name]

        if name in REQUIRED_KEYS:
            return state[name]

        return super().__getattribute__(name)

    def __getitem__(self, name):
        return self.__getattribute__(name)
    
    def __call__(self, *args, **kwargs):
        state = self._state
        next_state, output = state['apply'](state, self._global_config, *args, **kwargs)
        self._state = merge_trees(state, next_state, keys_to_merge=['params', 'aux'])
        return output
    

    def register_parameter(self, name, value):
        self._state['params'][name] = value

    def register_aux(self, name, value):
        self._state['aux'][name] = value

    def register_constant(self, name, value):
        self._state['constants'] = value

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
        elif is_jax_type(value):
            # by default we put things in params if they are jittable
            state['params'][name] = value
            return value
    

        return super().__setattr__(name, value)

