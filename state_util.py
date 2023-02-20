
from jax.tree_util import tree_map


def group_state_list(states):
    '''
    Extracts values for each key from a list of dicts and groups them together.

    arguments:
        states: an iterable of dict-like objects. This function is only 
            invertible if each entry of states has the same set of keys.
    
    returns:
        grouped: a dict whose keys are the union of all keys in elements
            of `states`, and whose values are lists of all corresponding values
            for all elements of `states`. That is, if all entries of states
            have the same keys, then grouped[key][i] = states[i][key].
    '''
    grouped = {}
    for state in states:
        for key in state:
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(state[key])



    return grouped

def ungroup_state(state):
    '''
    inverse of group_state_list when it is invertible.
    
    arguments:
        state: a dict-like object whose values are iterables all
            of the same length
        
    returns:
        ungrouped: a list of dict-like objects with the same keys as
            state such that ungrouped[i][key] = state[key][i].
    '''

    grouped = None

    for key in state:
        # initialized grouped in the first iteration
        if grouped is None:
            grouped = [{} for _ in state[key]]
        for i, value in enumerate(state[key]):
            grouped[i][key] = value

    return grouped


def tree_merge_none(*trees):
    '''
    takes a list of trees and produces a single tree.
    Each input tree must have the same shape.
    The output tree will also have the same shape.
    
    Each leaf of the output tree will be determined by the list of corresponding
    leaf values in `trees`: the value will simply be the first non-None value of this list.

    otherwise, 
     1. the first non-None, non-dictionary corresponding leaf value in `trees`.
     2. if the first non-None leaf value in `trees` is a dict, then
    '''

    def find_value(*values):
        for v in values:
            if v is not None:
                return v
                
        return None
    
    return tree_map(find_value, trees)

def tree_split_by_keys(tree, keys=['params', 'constants']):
    return [tree_extract_values(tree, key) for key in keys]




def tree_mask(to_mask, aux, mask_fn):
    '''
    creates a new tree by replacing leaves of `to_mask` with None.
    aux should be another tree whose shape is an extension of `to_mask`.
    That is to_mask should be a subtree of `aux`, so that each leaf of `to_mask`
    corresponds to a subtree of `aux`

    mask_fn is a function that takes the subtree of `aux` corresponding to a leaf of `to_mask`
    and returns a bool that is False if we should replace this leaf of `to_mask` with None.

    Common use-case:

    to_mask = {
        'fc': {
            'weight': [jnp. array]
            'bias': [jnp.array]
        }
        'const_buffer: [jnp.array]
    }

    aux = {
        'fc': {
            'weight': {
                'weight_decay': 0.001,
                'trainable': True
            },
            'bias': {
                'weight_decay': 0.0,
                'trainable': True
            }
        }
        'const_buffer': {
            'trainable': False
        }
    }

    mask_fn = lambda a: a['trainable']

    The returned value will be:

    {
        'fc': {
            'weight': [jnp. array]
            'bias': [jnp.array]
        }
        'const_buffer: None
    }

    Note that
    tree_merge_none(mask_tree(tree, ...), tree) is always equal to tree.

    '''

    return tree_map(lambda x, a: x if mask_fn(a) else None, to_mask, aux)


def split_trainable(state, aux):
    '''
    split a nn state tree into trainable and non-trainable params.
    '''

    return tree_mask(state, aux, lambda a: a['trainable']), mask_tree(state, aux, lambda a: not a['trainable'])
