# HAX
## A "simple" neural network library on top of JAX.

BU SCC setup instructions:
```
module load python3 pytorch tensorflow cuda/11.2 cudnn/8.1.1
# (set up/activate virtual env e.g. python -m venv gpujaxenv; source gpujaxenv/bin/activate)
pip install --upgrade pip
pip install -r requirements.txt
```


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
problems by providing utilities to directly compute with architectures described in a simple tree form. 

A Hax module is a pair consisting of a "structure tree" and a "global config". Both of these are simple python dictionaries. The global config should probably be even a
a simple JSON object of config values (e.g. {'training_mode': True}). The structure tree is a tree that contains both model weights and functions describing how to 
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
        return x, S

    rng = S["constants"]["rng"]
    rng, subkey = jax.random.split(rng)

    p = S["constants"]["p"]
    y = x * jax.random.bernoulli(rng, p, shape=x.shape)

    S["constants"]["rng"] = rng

    return S, y
```
Note that it is strongly advised NOT to change the `"apply"` or `"aux"` values of the input `S` inside these apply functions as this will cause
very weird behavior when jitting. Instead, these values are meant to be edited as part of an overall surgery on the model architecture. Several helper
functions provided by Hax (e.g. `bind_module`) explicitly enforce this behavior if they are used.



    




