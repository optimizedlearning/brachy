
import structure_util as su
from jax.tree_util import tree_map
from jax import numpy as jnp
import jax

def AdamW(model_state, lr=1.0, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8, params_filter=su.get_params):
    params, rest = params_filter(model_state)
    state = {
        'per_param_state': tree_map(
            lambda p: {
                'momentum_buffer': jnp.zeros_like(p),
                'variance_buffer': jnp.zeros_like(p)
            },
            params
        ),
        'beta1': jnp.array(betas[0]),
        'beta2': jnp.array(betas[1]),
        'lr_base': jnp.array(lr),
        'weight_decay': jnp.array(weight_decay),
        'eps': jnp.array(eps),
        't': jnp.array(0)
    }
    
    return state, AdamW_apply

def AdamW_apply(
    opt_state,
    model_state,
    example_data,
    value_and_grad_fn,
    lr=1.0,
    params_filter=su.get_params,
    params_merger=su.merge_trees):

    beta1 = opt_state['beta1']
    beta2 = opt_state['beta2']
    lr = opt_state['lr_base'] * lr
    weight_decay = opt_state['weight_decay']
    eps = opt_state['eps']

    opt_state['t'] += 1
    t = opt_state['t']

    (model_state_next, *value), grad = value_and_grad_fn(model_state, example_data)

    params, rest = params_filter(model_state_next)


    def per_param_state_update(g, state):
        m = state['momentum_buffer']
        v = state['variance_buffer']

        return {
            'momentum_buffer': m * beta1 +  (1.0-beta1) * g,
            'variance_buffer': v *beta2 +  (1.0-beta2)*g**2
        }
    
    # be careful: it is important that grad is the first argument after the update function here!
    opt_state['per_param_state'] = tree_map(
        per_param_state_update,
        grad,
        opt_state['per_param_state']
    )

    def update(p, state):
        m = state['momentum_buffer']/(1.0 - beta1**t)
        v = state['variance_buffer']/(1.0 - beta2**t)
        return p - lr * (m/(eps + jnp.sqrt(v)) + weight_decay * p)
    
    # be careful: it is important that params is the first argument after the update function here!
    params = tree_map(
        update,
        params,
        opt_state['per_param_state']
    )

    model_state = params_merger(rest, params)

    return opt_state, model_state, *value



    