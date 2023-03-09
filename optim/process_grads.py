import jax
from jax import numpy as jnp
from jax.tree_util import tree_map


def _per_coordinate_clip(x, value):
    return jnp.clip(x, a_min=-value, a_max=value)


def clip_grads(opt_state_apply, clip_value=1.0, clip_type='per_coordinate'):
    opt_state, opt_apply = opt_state_apply
    supported_clip_types = ['per_coordinate']
    if clip_type not in supported_clip_types:
        raise ValueError(f"unsupported clip_type: {clip_type}. Supported types are: {supported_clip_types}")

    if 'grad_clipping' not in opt_state:
        opt_state['grad_clipping'] = {}
    opt_state['grad_clipping'][clip_type] = clip_value

    if clip_type == 'per_coordinate':
        clip_fn = _per_coordinate_clip

    def new_opt_apply(
        opt_state,
        model_state,
        example_data,
        value_and_grad_fn,
        *args,
        **kwargs):
        clipping = opt_state['grad_clipping']

        def new_value_and_grad_fn(m_state, example_data):
            value, grad = value_and_grad_fn(m_state, example_data)
            grad = tree_map(lambda g: clip_fn(g, clipping[clip_type]), grad)
            return value, grad

        return opt_apply(
            opt_state,
            model_state,
            example_data,
            new_value_and_grad_fn,
            *args,
            **kwargs)

    return opt_state, new_opt_apply



