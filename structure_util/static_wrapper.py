
from jax.tree_util import tree_flatten, tree_map, tree_reduce
import structure_util as su
import jax
import inspect
from functools import wraps

from jax.core import valid_jaxtype

from typing import (Any, Callable, Generator, Hashable, Iterable, List, Literal,
                    NamedTuple, Optional, Sequence, Tuple, TypeVar, Union,
                    overload, cast)

_CACHED_WRAPS = {}


class HashableTree:
    def __init__(self, tree):
        self.tree = tree
        # we save the current hash of the tree here, so that future in-place changes to the tree
        # which might change the hash cannot fool us...
        self.hash = self.hash_tree()
    
    def hash_tree(self):
        leaves, treedef = tree_flatten(self.tree)
        leaves = tuple(leaves)
        return hash((leaves, treedef))

    def __hash__(self):
        # add 1 to the hash so that nobody sneaky can construct the following tuple
        # to engineer a collision (probably not a big deal, but anyway...)
        return hash((self.hash, self.hash_tree())) + 1
    
    def __eq__(self, other):
        # technically, this equality could return False for trees that are
        # really the same if they started out different and then were
        # modified in place to become the same.
        # However, a false negative will only result in an unecessary recompilation
        # while a false positive will result in reusing an incorrect cached jitted function.

        if not isinstance(other, HashableTree):
            return False
        eq = self.hash == other.hash
        if not eq:
            return False
        try:
            eq = tree_reduce(lambda a,b: a and b, tree_map(lambda x,y: x==y, self.tree, other.tree), True)
        except:
            return False
        
        return eq

    def __str__(self):
        return f"HashableTree<{self.tree}>"


# These are a bit hacky...
def split_jax_nonjax(tree):
    keep_static = lambda x: not valid_jaxtype(x) and isinstance(x, Hashable)
    jaxtype_tree = tree_map(lambda x: None if keep_static(x) else x, tree)
    nonjaxtype_tree = tree_map(lambda x: x if keep_static(x) else None, tree)

    return jaxtype_tree, nonjaxtype_tree

def merge_jax_nonjax(jax_tree, nonjax_tree):
    def merge(jt, njt):
        if njt == None:
            return jt
        else:
            return njt

    return tree_map(merge, jax_tree, nonjax_tree, is_leaf=lambda x: x is None)


def improved_static(wrapper, *outer_args, static_argnums=None , static_argnames=None, **outer_kwargs):

    wrapper_signature = inspect.signature(wrapper)
    wrapper_parameters = wrapper_signature.parameters
    wrapper_paramnames = list(wrapper_parameters.keys())

    if wrapper not in _CACHED_WRAPS:
        _CACHED_WRAPS[wrapper] = {}

    _CACHED_FUNCS = _CACHED_WRAPS[wrapper]

    # outer_static_argnums = static_argnums
    # outer_static_argnames = static_argnames

    # this decorator actually wraps a decorator (the argument "wrapper") itself, so we must return a decorator.
    # this function static_wrapper is what we will return.
    @wraps(wrapper)
    def static_wrapper(fun, *wrapper_args, static_argnums=None , static_argnames=None, **wrapper_kwargs):

        # # Some checks to allow for default arguments specified in a decorator...
        # # this might be overly complicated a feature to have...
        # if outer_static_argnums is not None:
        #     assert static_argnums is None, "ambiguous setting for static_argnums in wrapper {fun}!"
        #     static_argnums = outer_static_argnums

        # if outer_static_argnames is not None:
        #     assert static_argnames is None, "ambiguous setting for static_argnames in wrapper {fun}!"
        #     static_argnames = outer_static_argnames

        # if len(outer_args) == 0:
        #     assert len(wrapper_args) == 0, "ambiguous args for wrapper {fun}!"
        #     wrapper_args = outer_args

        # for k,v in outer_kwargs.items():
        #     assert k not in wrapper_kwargs, "ambiguous kwargs for wrapper {fun}!"
        #     wrapper_kwargs[k] = v


        # check if static_argnums or static_argnames are specified as position arguments.
        if 'static_argnums' in wrapper_paramnames and len(wrapper_args) >= wrapper_paramnames.index('static_argnums'):
            static_argnums = wrapper_args[wrapper_argnames.index('static_argnums')]
        if 'static_argnames' in wrapper_paramnames and len(wrapper_args) >= wrapper_paramnames.index('static_argnames'):
            static_argnames = wrapper_args[wrapper_paramnames.index('static_argnames')]

        # canonicalize static_argnums and static_argnames as lists.
        if isinstance(static_argnums, int):
            static_argnums = [static_argnums]
        if isinstance(static_argnames, str):
            static_argnames = [static_argnames]
        
        if static_argnums is None:
            static_argnums = []
        if static_argnames is None:
            static_argnames = []

        static_argnums = list(static_argnums)
        static_argnames = list(static_argnames)


        # get information about the function we are going to wrap.
        signature = inspect.signature(fun)
        parameters = signature.parameters
        parameter_list = list(parameters.keys())

        # canonicalize static_argnums and static_argnames some more: make them
        # refer to the same set of arguments as much as possible to maximize the
        # number of cache hits later.
        for name in static_argnames:
            num = parameter_list.index(name)
            if num not in static_argnums:
                if parameters[name].kind != parameters[name].KEYWORD_ONLY:
                    static_argnums.append(num)
        
        for num in static_argnums:
            name = parameter_list[num]
            if name not in static_argnames:
                if parameters[name].kind != parameters[name].POSITIONAL_ONLY:
                    static_argnames.append(name) 

        # initialize cache for this function.
        if fun not in _CACHED_FUNCS:
            _CACHED_FUNCS[fun] = {}

        cached_calls = _CACHED_FUNCS[fun]
        
        # this is the function that we will actually return from this decorator.
        # it first computes all static arguments and checks a cache to see if this
        # function has been called with these static arguments before. If not,
        # then it calles the base wrapper (i.e. jax.jit) on a version of the  base function
        # to wrap that has all the static arguments included via lexical capture.
        # Otherwise, it looks up this wrapped function in the cache and calls it.
        # Finally, we process the outputs of the function to add back in non-jaxtype outputs.
        # It is assumed that if the static arguments are the same and the non-static arguments
        # have the same shape, then the non-jaxtype outputs are also the same.
        @wraps(fun)
        def wrapped_fun(*args, **kwargs):
            
            # process the args to extract static arguments  and prepare the
            # cache key.
            split_args = []
            structure_tree_args_statics = {}
            structure_tree_kwargs_statics = {}

            tree_args_statics = {}
            tree_kwargs_statics = {}
            for argnum, arg in enumerate(args):
                if not su.is_structure_tree(arg):

                    jax_tree, nonjax_tree = split_jax_nonjax(arg)
                    split_args.append(jax_tree)
                    tree_args_statics[argnum] = nonjax_tree
                else:
                    params_buffers, rest = su.split_non_static(arg)
                    split_args.append(params_buffers)
                    structure_tree_args_statics[argnum] = rest        

            split_kwargs = {}
            for k, v in kwargs.items():
                if not su.is_structure_tree(v):
                    jax_tree, nonjax_tree = split_jax_nonjax(v)
                    split_kwargs[k] = jax_tree
                    tree_kwargs_statics[k] = nonjax_tree
                else:
                    params_buffers, rest = su.split_non_static(v)
                    split_kwargs[k] = params_buffers
                    structure_tree_kwargs_statics[k] = rest  


            static_args = [arg if i in static_argnums else None for i, arg in enumerate(split_args)]
            static_kwargs = {
                k: split_kwargs.get(k) for k in static_argnames
            }

            cache_key = HashableTree({
                'static_argnums': static_argnums,
                'static_args': static_args,
                'static_argnames': static_argnames,
                'static_kwargs': static_kwargs,
                'structure_tree_args_statics': structure_tree_args_statics,
                'tree_args_statics': tree_args_statics,
                'structure_tree_kwargs_statics': structure_tree_kwargs_statics,
                'tree_kwargs_statics': tree_kwargs_statics,
            })

            # cache miss - define a function to wrap with the base wrapper.
            if cache_key not in cached_calls:
                def to_wrap(*args, **kwargs):
                    args_with_statics = list(args)
                    for i in range(len(args)):
                        if i in static_argnums:
                            args_with_statics[i] = static_args[i]
                        if i in structure_tree_args_statics:
                            args_with_statics[i] = su.merge_trees(args_with_statics[i], structure_tree_args_statics[i])
                        if i in tree_args_statics:
                            args_with_statics[i] = merge_jax_nonjax(args_with_statics[i], tree_args_statics[i])
                    
                    kwargs_with_statics = dict(kwargs)
                    for k in kwargs:
                        if k in static_argnames:
                            kwargs_with_statics[k] = static_kwargs[k]
                        if k in structure_tree_kwargs_statics:
                            kwargs_with_statics[k] = su.merge_trees(kwargs_with_statics[k], structure_tree_args_statics[k])
                        if k in tree_kwargs_statics:
                            kwargs_with_statics[k] = merge_jax_nonjax(kwargs_with_statics[k], tree_kwargs_statics[k])
                    values = fun(*args_with_statics, **kwargs_with_statics)

                    if not isinstance(values, tuple):
                        values = [values]
                    
                    # this should happen upon first tracing to populate the static parts of any structure trees
                    # in the returned values.
                    returned_structure_statics = {}
                    split_values = list(values)
                    for i, v in enumerate(values):
                        jax_tree, nonjax_tree = split_jax_nonjax(v)
                        if cached_calls[cache_key]['returned_structure_statics'] is None:
                            returned_structure_statics[i] = nonjax_tree
                        split_values[i] = jax_tree
                    if cached_calls[cache_key]['returned_structure_statics'] is None:
                        cached_calls[cache_key]['returned_structure_statics'] = returned_structure_statics
                    if len(split_values) == 1:
                        return split_values[0]
                    else:
                        return tuple(split_values)

                # add this wrapped function to the cache.
                cached_calls[cache_key] = {}
                cached_calls[cache_key]['wrapped_func'] = wrapper(to_wrap, *wrapper_args, **wrapper_kwargs)
                cached_calls[cache_key]['returned_structure_statics'] = None

            wrapped = cached_calls[cache_key]['wrapped_func']

            args_without_statics  = list(split_args)
            kwargs_without_statics  = dict(split_kwargs)


            for i in static_argnums:
                if i < len(args):
                    args_without_statics[i] = None
            for k in static_argnames:
                if k in kwargs:
                    kwargs_without_statics[k] = None
                     
            # add back cached static outputs to the jaxtype outputs.
            values = wrapped(*args_without_statics, **kwargs_without_statics)

            if not isinstance(values, tuple):
                values = [values]
            values = list(values)
            
            for i, v in cached_calls[cache_key]['returned_structure_statics'].items():
                values[i] = merge_jax_nonjax(values[i], v)
            
            if len(values) == 1:
                return values[0]
            else:
                return tuple(values)


        return wrapped_fun

    return static_wrapper


jit = improved_static(jax.jit)
