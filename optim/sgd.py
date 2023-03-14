
import structure_util as su
from jax.tree_util import tree_map
from jax import numpy as jnp
import jax

def SGD(model_tree, lr=1.0, momentum=0.0, weight_decay=0.0, params_filter=su.get_params, params_merger=su.merge_trees):
    organizer = su.StateOrganizer()
    params, rest = params_filter(model_tree)
    organizer.register_buffer(
        'momentum_buffer', tree_map(lambda x: jnp.zeros_like(x), params)
    )
    organizer.momentum_coef = momentum
    organizer.weight_decay = weight_decay
    organizer.register_aux('params_filter', params_filter)
    organizer.register_aux('params_merger', params_merger)

    organizer.update_global_config('lr', lr)
    
    return organizer.create_module(SGD_apply)

# for now, we assume the the first return value of value_and_grad_fn
# is the model update. Maybe in future we allow this to be configurable...
def SGD_apply(
    opt_tree,
    opt_config,
    value_and_grad_fn,
    *value_grad_args,
    **value_grad_kwargs
    ):
    organizer = su.StateOrganizer(opt_tree, opt_config)

    momentum_buffer = organizer.momentum_buffer
    momentum_coef = organizer.momentum_coef

    lr = opt_config['lr']

    weight_decay = organizer.weight_decay

    (model, *value), grad = value_and_grad_fn(*value_grad_args, **value_grad_kwargs)

    params, rest = organizer.params_filter(model)

    momentum_buffer_next = tree_map(
        lambda m, g, p: m * momentum_coef +  (g + weight_decay * p),
        momentum_buffer, grad, params
    )

    organizer.momentum_buffer = momentum_buffer_next

    params = tree_map(
        lambda p, m: p - lr * m,
        params, momentum_buffer_next
    )

    updated_model = organizer.params_merger(rest, params)

    return organizer.get_state(), updated_model, *value



    