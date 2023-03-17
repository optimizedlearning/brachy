# Brachy
## A (increasingly less) simple neural network library on top of JAX.

## Better JIT wrapper
First, it is very annoying that `jax.jit` cannot handle dictionaries as static arguments, or arguments that are pytree where some values are static and some can be traced, or functions that return static values. So, we provide a general wrapper in `structure_util.improved_static` that takes care of this by automatically separating  out traceable and non-traceable components of arguments before 
passing to `jit`. It also can handle non-jaxtypes in return values. However, be careful with these: we assume that any non-jaxtype in a return value must be a fixed function of
the static arguments and the shape of the traced arguments (i.e. their values do not change unless the function needs to be re-traced).
```
import jax
from jax import numpy as jnp
from brachy import structure_util as su

jit  = su.jit
# su.jit is an alias for su.improved_static(jax.jit).

@jit
def foo(x,y):
    if x['q'] == 'go ahead!':
        return {'a': x['a'], 'b': y['b']}
    else:
        return {'a': 2*y['a'], 'b': y['b']}

x = {
    'q': 'stop',
    'a': jnp.ones(3)
}
y = {
    'a': jnp.ones(5),
    'b': ['hello', 'friend']
}

z = foo(x,y) 
## z should be:
# {'a': jnp.array([2,2,2,2,2]), 'b': ['hello', 'friend']}
```

Further, this wrapper also will automatically extract and make static the static components of a structure tree (described below). That is,
if a structure tree has an otherwise traceable value under the `'aux'` key (e.g. a `False` for configuration or similar), then it will not be traced.

This wrapper `su.improved_static` can also  be used  to impart similar behavior to other JAX primitives (e.g. `xmap`).


## Overview
HAX tries to keep your code as close to the functional spirit of JAX as possible
while also facilitating easy portability from pytorch.


In pytorch, a module is an instance of a class that stores various model parameters
(i.e. network weights) as class members. These are relatively straightforward to code up, but have two
important drawbacks. First, this style of module does not play nice with JAX's functional programming style.
This means that it is difficult to implement more complicated ideas such as meta-gradient descent (i.e. differentiating with respect to hyperparameters).
Second, as models grow in size and complexity, it will likely become more and more important to be able to "mix and match" different
components of pre-trained modules in an easy way. Right now, to extract the output of some intermediate layer or to add a new layer somewhere
in the module computation requires a careful inspection of the source code and often some extra work to transfer pretrained weights to the new architecture.
However, this is not really necessary: model architectures are usually relatively straightforwardly described as simple trees. Hax exploits this to solve both
problems by providing utilities to directly compute with architectures described in a tree form. 

A Hax module is a pair consisting of a "structure tree" and a "global config". Both of these are python dictionaries. The global config should probably be even a
a JSON object of config values (e.g. {'training_mode': True}). The structure tree is a tree that contains both model weights and functions describing how to 
apply these weights. We *could* have tried to organize the structure tree as a python class. However, we wanted to make the structure trees as hackable as possible. Wrapping them in some complicated class mechanism in order to provide some ease of use in common cases might make this more difficult. That said, Hax does still provide a class `StateOrganizer` that can be used to convert a structure tree into a class that behaves very similarly to a pytorch module, which is useful for building structure trees.

Formally, a Hax structure tree `S` is a `dict` whose keys are  `"params"`, `"buffers"`, `"aux"`, `"apply"`, and `"submodules"`.
The value `S["submodules"]` is either a dict whose values are themselves structure trees (i.e. `S["submodules"]` specified the children of `S` 
in the tree).
The values `S["params"]` and `S["buffers"]` are both dicts whose values are *JAX types*. By a JAX type, we mean a value that is a valid argument
to a traced JAX functions (e.g. a pytree where all leaves are JAX arrays). That is, the function:
```
@jax.jit
def identity(x):
    return jax.tree_utils.tree_map(lambda a:a, x)
```
will run without error on any JAX type.

The value `S["apply"]` is a function with signature:
```
def apply(
    structure_tree: Hax.structure_tree,
    global_config: dict,
    *args,
    **kwargs) -> Hax.structure_tree, Any
```
`Hax.structure_tree` is simply an alias for a dict, so any function that takes a dict as the first two arguments
and returns a dict is acceptable. The additional arguments to `apply` will be implementation specific. The first
return value is the "output" of the module, and the second return value is an updated version of the
input argument `structure_tree`. For example, a linear layer might be implemented as follows:

```
def linear_apply(S: Hax.structure_tree, global_config: dict, x: Array) -> Array, Hax.structure_tree:
    weight = S["params"]["weight"]
    bias = S["params"]["bias"]

    y = x @ weight + bias

    return S, y
```

In this case, we did not need to change the input structure tree. However, layers that require state, randomization, or different
behaviors in the train or eval setting require more delicate construction:

```
def dropout_apply(S: Hax.structure_tree, global_config: dict, x: Array) -> Array, Hax.structure_tree:
    if not global_config["is_training"]:
        return S, x

    rng = S["buffers"]["rng"]
    rng, subkey = jax.random.split(rng)

    p = S["buffers"]["p"]
    y = x * jax.random.bernoulli(subkey, p, shape=x.shape)/p

    S["buffers"]["rng"] = rng

    return S, y
```
Note that it is strongly advised NOT to change the `"apply"` or `"aux"` values of the input `S` inside these apply functions as this will cause
retracing when jitting. Instead, these values are meant to be edited as part of an overall surgery on the model architecture.

## Structure Tree Terminology

Technically, many of the functions in this package do not require a structure tree to have all the keys `"params"`, `"buffers"`, `"aux"`, `"apply"`: only the `"submodules"` key is really needed. Given a structure tree `tree`, we say that `tree` is a leaf if `tree["submodules"] = {}`. Further, we say that `tree` is a node with path `[k1, k2, k3]` if there is a root tree `root` such that `tree = root["submodules"][k1]["submodules"][k2]["submodules"][k3]`. In general, the path of `tree["submodules"][k]` is `P + [k]` where `P` is the path of `tree`.

## Structure Tree Utils

`brachy.structure_tree_util` contains the core functions that power converting structure trees into the forward pass function for a module and back.
Key utilities include:

* `structure_tree_map(func: Callable, *trees: List[dict], path={}) -> Union[dict, Tuple[dict,...]]`. The first argument is a function `func(*nodes, path)` that outputs a leaf node (or a tuple of leaf nodes).The second argument `trees` must be either a single structure tree or a list of structure trees. The output will be a structure tree such that for each unique path `P` in any of the trees in `trees`, the output tree will have a node with path `P` that is the output of `func` with first argument `nodes` being the list `[subtree of tree at path P for tree in trees]` and `path=P`. If `func` returns multiple trees, then `structure_tree_map` will output the corresponding multiple trees.
* `StateOrganizer`: this class does a lot of the heavy lifting to make defining new structure trees similar to the experience of defining a module in pytorch. Eventually, one can call `organizer.create_module()` to obtain a tuple `tree, global_config`. When building the tree, if you assign a new attribute to a `StateOrganizer` object with a tuple via `'organizer.name = (subtree, sub_global_config)`, then the tree returned by `organizer.create_module()` will have `subtree` as the value `["submodules"][name]`. Also, `global_config` will be merged with `sub_global_config` (value in `sub_global_config` do not override old values).
See the examples directory to see how to use `StateOrganizer` objects.
* `apply_tree(tree: dict, global_config: dict, *args, **kwargs)`. This function is a shorthand for `tree['apply'](tree, global_config, *args, **kwargs)`.
* `bind_module(tree: dict, global_config: dict) -> dict, Callable`. This function is mostly unecessary given the updated `brachy.structure_util.jit` functionality described earlier. It takes as input a structure tree and a global config and returns a `state` dictionary and an `apply` function. The state dictionary is just the original structure tree with all but the `"params"`, `"buffers"`, and `"submodules"` keys removed. This represents the current mutable state of the module. The apply function will apply the tree: it takes a state dictionary and whatever inputs the module requires and returns both an updated state dictionary and all the ouptuts of the module.
The returned `apply` function from `bind_module` can be Jitted as it captures the unhashable `global_config` dictionary in a closure. To change the global config dictionary, use `apply.bind_global_config(new_global_config)`. To recover a full structure tree, use `tree, global_config = unbind_module(state, apply)`.

## Random number generator utils

The file `rng_util.py` contains a context manager that makes it easier to pass JAX prngkeys down through a tree of functions without having to write a ton of `rng, subkey = jax.random.split(rng)` all over the place. See the comments at the top of the file or the usages in the resnet example or the `nn.py` file for more info.

This utility can be combined with the `StateOrganizer` via the decorators `organized_init_with_rng` and `organized_apply` defined in `structure_util.py`. See the language modeling example for these decorators in use.
    

## Installing

### From pip
You can now `pip install brachy`! However, this will explicitly NOT install jax as the installation process for jax seems to differ depending on GPU vs CPU. You should install the appropriate jax version


### BU SCC setup instructions
You need python3, and jax (pytorch useful for dataloaders, or running tests). Currently there seems to be some issue preventing simultaneous loading of jax, pytorch and tensorflow. However, we probably don't need tensorflow so it is not a huge problem.
```
module load python3 pytorch cuda/11.6 jax/0.4.6
```
You should probably also setup a virtual environment: `python -m venv brachyenv` to create, `source brachyenv/bin/activate` to activate,  `deactive` to leave the environment.

Some of the example require additional packages listed in the `requirements.txt` file. You can `pip install --upgrade pip` and then `pip install -r requirements.txt` to get them to run.
Or just run an example and then do `pip  install` one by one as you get "ModuleNotFoundError



