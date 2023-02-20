# HAX
## A "simple" neural network library on top of JAX.


HAX tries to keep your code as close to the functional spirit of JAX as possible
while also facilitating easy portability from pytorch.

In pytorch, a module is an instance of a class that stores various model parameters
(i.e. network weights) as class members. In contrast, a JAX code does not like objects
with state. So, in HAX, the equivalent of a pytorch module is a *pair* `state`, `apply`
such that `state` is a pytree object containing model state and `apply` is a function
whose first argument is  `state` and subsequent arguments represent inputs to the network.

The goal is to keep things as close as possible to simple JAX. We don't even go as far as to 
store the `state` of a module in a `module.state` variable or the `apply` function in a `module.apply` function.
Doing so we feel encourages treating these as actual state, which can make performing complicated transformations
even more complicated.

Hax provides tools to help build these pairs. The tools do impose some mild limitations on the structure of
the `state` pytrees. You do not have to use them! These tools work by describing a module through a concept called a **structure tree**.

A Hax structure tree `H` is simply a dict containing the key `__apply__`. The value `H["__apply__"]` is a function. All other values must either be `None` or *also* be a structure tree. We say that such values are "children" of `H`. The descendents of `H` is the set of structure trees containing `H` as well as any structure tree that is a child of a descendent of `H`. A pytree `T` is **compatible** with a structure tree `H` if the subtree of `H` formed by recursively deleting all `__apply__` from all descendents of `H` is a subtree of `T`. The first argument to the function `H["__apply__"]` should be a pytree that is compatible with `H`.

A structure tree describes the topology of a neural network in a precise way that allows for 
