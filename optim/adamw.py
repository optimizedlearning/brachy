
import structure_util as su
from jax.tree_util import tree_map
from jax import numpy as jnp
import jax

def AdamW(model_tree, lr=1.0, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8, params_filter=su.get_params, params_merger=su.merge_trees):
    organizer = su.StateOrganizer()
    organizer.betas = betas
    organizer.weight_decay = weight_decay
    organizer.register_buffer('eps', eps)
    organizer.register_buffer('t', 0)

    organizer.lr = lr

    params, rest = params_filter(model_tree)

    organizer.register_buffer(
        'per_variable_state',
        tree_map(
            lambda p: {
                'momentum': jnp.zeros_like(p),
                'variance': jnp.zeros_like(p)
            },
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
    opt_tree: dict,
    opt_config: dict,
    hparams: dict,
    value_and_grad_fn: callable,
    *value_grad_args,
    **value_grad_kwargs):

    organizer = su.StateOrganizer(opt_tree, opt_config)

    (model, *value), grad = value_and_grad_fn(*value_grad_args, **value_grad_kwargs)

    params, rest = organizer.params_filter(model)


    organizer.t = organizer.t + 1
    t = organizer.t
    beta1 = organizer.betas[0]
    beta2 = organizer.betas[1]

    lr = organizer.lr
    if 'lr' in hparams:
        lr *= hparams['lr']

    weight_decay = organizer.weight_decay
    eps = organizer.eps


    organizer.per_variable_state = tree_map(
        lambda g, s: {
            'momentum': s['momentum']*beta1 + (1.0-beta1) * g,
            'variance': s['variance']*beta2 + (1.0-beta2) * g**2
        },
        grad,
        organizer.per_variable_state
    )


    def update(p, state):
        m = state['momentum']
        v = state['variance']
        m_hat = m/(1.0 - beta1**t)
        v_hat = v/(1.0 - beta2**t)
        return p - lr * (m_hat/(eps + jnp.sqrt(v_hat)) + weight_decay * p)
    
    # be careful: it is important that params is the first argument after the update function here!
    params = tree_map(
        update,
        params,
        organizer.per_variable_state
    )

    updated_model = organizer.params_merger(rest, params)

    return organizer.get_state(), updated_model, *value



    