
import dill
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten
from jax import numpy as jnp
from jax import Array

    
def save(tree, global_config, fp):

    leaves, treedef = tree_flatten(tree)

    array_leaves = [x if isinstance(x, Array) else None for x in leaves]
    nonarray_leaves = [None if isinstance(x, Array) else x for x in leaves]

    dill.dump(['haxcheckpointversion', 1], fp)
    dill.dump(global_config, fp)
    dill.dump(nonarray_leaves, fp)
    dill.dump(treedef, fp)

    for i, array in enumerate(array_leaves):
        if array is not None:
            dill.dump(i, fp)
            jnp.save(fp, array)

    dill.dump(None, fp)


def load(fp):

    version_info = dill.load(fp)
    assert(version_info[0] == 'haxcheckpointversion')
    assert(version_info[1] == 1)

    global_config = dill.load(fp)

    leaves = dill.load(fp)
    treedef = dill.load(fp)

    index = dill.load(fp)

    while index is not None:
        array = jnp.load(fp)
        leaves[index] = array
        index = dill.load(fp)
    
    tree = tree_unflatten(treedef, leaves)

    return tree, global_config
    
    

    