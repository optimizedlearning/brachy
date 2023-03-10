
import structure_util as su
from jax.tree_util import tree_map
from jax import numpy as jnp
import jax

def SGD(model, lr=1.0, momentum=0.0, weight_decay=0.0, params_filter=su.get_params, params_merger=su.merge_trees):
    organizer = su.StateOrganizer()
    organizer.model_to_optimize = model

    params, rest = params_filter(model[0])
    organizer.register_buffer(
        'momentum_buffer', tree_map(lambda x: jnp.zeros_like(x), params)
    )
    organizer.momentum_coef = momentum
    organizer.lr_base = lr
    organizer.weight_decay = weight_decay
    organizer.register_aux('params_filter', params_filter)
    organizer.register_aux('params_merger', params_merger)
    
    return organizer.create_module(SGD_apply)


def SGD_apply(
    tree,
    global_config,
    example_data,
    value_and_grad_fn,
    lr=1.0):
    organizer = su.StateOrganizer(tree, global_config)

    momentum_buffer = organizer.momentum_buffer
    momentum_coef = organizer.momentum_coef
    lr = organizer.lr_base * lr
    weight_decay = organizer.weight_decay

    (model_update, *value), grad = value_and_grad_fn(organizer.model_to_optimize.get_tree(), global_config, example_data)

    params, rest = organizer.params_filter(model_update)

    momentum_buffer_next = tree_map(
        lambda m, g, p: m * momentum_coef +  (g + weight_decay * p),
        momentum_buffer, grad, params
    )

    organizer.momentum_buffer = momentum_buffer_next

    params = tree_map(
        lambda p, m: p - lr * m,
        params, momentum_buffer_next
    )

    model_update = organizer.params_merger(rest, params)

    organizer.model_to_optimize.update_tree(model_update)

    return organizer.get_state_update(), *value



    