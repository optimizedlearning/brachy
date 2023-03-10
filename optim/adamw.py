
import structure_util as su
from jax.tree_util import tree_map
from jax import numpy as jnp
import jax

def AdamW(model, lr=1.0, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8, params_filter=su.get_params, params_merger=su.merge_trees):
    organizer = su.StateOrganizer()
    organizer.model_to_optimize = model
    organizer.base_lr = lr
    organizer.betas = betas
    organizer.weight_decay = weight_decay
    organizer.register_buffer('eps', eps)
    organizer.register_buffer('t', 0)

    params, rest = params_filter(model[0])

    organizer.register_buffer(
        'momentum',
        tree_map(
            lambda p: jnp.zeros_like(p),
            params
        )
    )

    organizer.register_buffer(
        'variance',
        tree_map(
            lambda p: jnp.zeros_like(p),
            params
        )
    )

    organizer.register_aux(
        'params_filter', params_filter
    )
    organizer.register_aux(
        'params_merger', params_merger
    )

    return organizer.create_module(AdamW_apply)


def AdamW_apply(
    opt_tree,
    global_config,
    example_data,
    value_and_grad_fn,
    lr=1.0):

    organizer = su.StateOrganizer(opt_tree, global_config)

    (model_update, *value), grad = value_and_grad_fn(organizer.model_to_optimize.get_tree(), global_config, example_data)

    params, rest = organizer.params_filter(model_update)


    organizer.t = organizer.t + 1
    t = organizer.t
    beta1 = organizer.betas[0]
    beta2 = organizer.betas[1]
    lr *= organizer.base_lr
    weight_decay = organizer.weight_decay
    eps = organizer.eps

    # be careful: it is important that grad is the first argument after the update function here!
    organizer.momentum = tree_map(
        lambda g, m: m*beta1 + (1.0-beta1) * g,
        grad,
        organizer.momentum
    )

    organizer.variance = tree_map(
        lambda g, v: v*beta2 + (1.0-beta2) * g**2,
        grad,
        organizer.variance
    )

    def update(p, m, v):
        m_hat = m/(1.0 - beta1**t)
        v_hat = v/(1.0 - beta2**t)
        return p - lr * (m_hat/(eps + jnp.sqrt(v_hat)) + weight_decay * p)
    
    # be careful: it is important that params is the first argument after the update function here!
    params = tree_map(
        update,
        params,
        organizer.momentum,
        organizer.variance
    )

    model_update = organizer.params_merger(rest, params)
    organizer.model_to_optimize.update_tree(model_update)

    return organizer.get_state_update(), *value



    