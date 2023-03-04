
import structure_util as su
from jax.tree_util import tree_map
from jax import numpy as jnp
import jax

def SGD(model_state, lr_base=1.0, momentum=0.0, weight_decay=0.0, params_filter=su.get_params):
    params = params_filter(model_state)
    state = {
        'momentum_buffer': tree_map(lambda x: jnp.zeros_like(x), params),
        'momentum_coef': momentum,
        'lr_base': lr_base,
        'weight_decay': weight_decay,
    }
    
    return state, SGD_apply

def SGD_apply(
    sgd_state,
    value_and_grad_fn,
    model_state,
    lr=1.0,
    params_filter=su.get_params,
    params_merger=su.merge_trees):

    momentum_buffer = sgd_state['momentum_buffer']
    momentum_coef = sgd_state['momentum_coef']
    lr = sgd_state['lr_base'] * lr
    weight_decay = sgd_state['weight_decay']

    (model_state_next, value), grad = value_and_grad_fn(model_state)

    params = params_filter(model_state_next)



    momentum_buffer_next = tree_map(
        lambda m, g, p: m * momentum_coef + (1.0 - momentum_coef) * (g + weight_decay * p),
        momentum_buffer, grad, params
    )
    sgd_state['momentum_buffer'] = momentum_buffer_next

    params = tree_map(
        lambda p, m: p - lr * m,
        params, momentum_buffer_next
    )
    # print("momentum buffer: ",momentum_buffer_next)

    model_state = params_merger(model_state_next, params)

    return model_state, sgd_state, value



    