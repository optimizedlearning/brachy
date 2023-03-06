
import structure_util as su
from jax.tree_util import tree_map
from jax import numpy as jnp
import jax

def AdamW(model_state, lr=1.0, beta1=0.9, beta2=0.99, weight_decay=0.0, eps=1e-5, params_filter=su.get_params):
    params = params_filter(model_state)
    state = {
        'momentum_buffer': tree_map(lambda x: jnp.zeros_like(x), params),
        'beta1': beta1,
        'beta2': beta2,
        'variance_buffer': tree_map(lambda x: jnp.zeros_like(x), params),
        'lr_base': lr,
        'weight_decay': weight_decay,
        'eps': eps
    }
    
    return state, AdamW_apply

def AdamW_apply(
    opt_state,
    model_state,
    value_and_grad_fn,
    lr=1.0,
    params_filter=su.get_params,
    params_merger=su.merge_trees):

    momentum_buffer = opt_state['momentum_buffer']
    beta1 = opt_state['beta1']
    beta2 = opt_state['beta2']
    variance_buffer = opt_state['variance_buffer']
    lr = opt_state['lr_base'] * lr
    weight_decay = opt_state['weight_decay']
    eps = opt_state['eps']

    (model_state_next, *value), grad = value_and_grad_fn(model_state)

    params = params_filter(model_state_next)

    momentum_buffer_next = tree_map(
        lambda m, g: m * beta1 +  (1.0-beta1) * g,
        momentum_buffer, grad
    )
    opt_state['momentum_buffer'] = momentum_buffer_next

    variance_buffer_next = tree_map(
        lambda v, g: v *beta2 +  (1.0-beta2)*g**2,
        variance_buffer, grad
    )
    opt_state['variance_buffer'] = variance_buffer_next

    params = tree_map(
        lambda p, m, v: p - lr * (m/(eps + jnp.sqrt(v)) + weight_decay * p),
        params, momentum_buffer_next, variance_buffer_next
    )

    model_state = params_merger(model_state_next, params)

    return opt_state, model_state, *value



    