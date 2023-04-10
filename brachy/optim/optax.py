
try:
    import optax
except:
    optax = None

from brachy import structure_util as su

def init(optax_optim, model_tree, *args, params_filter=su.get_params, params_merger=su.merge_trees, **kwargs,):
    organizer = su.StateOrganizer()
    params, rest = params_filter(model_tree)
    opt_state = optax_optim.init(params, *args, **kwargs)
    organizer.register_buffer(
        'opt_state', opt_state
    )
    organizer.register_static(
        'opt_apply', optax_optim.update
    )

    organizer.register_static(
        'params_filter', params_filter
    )
    organizer.register_static(
        'params_merger', params_merger
    )
    
    return organizer.create_module(optax_apply)



def optax_apply(
    opt_tree: dict,
    opt_config: dict,
    hparams: dict,
    value_and_grad_fn: callable,
    model_tree: dict,
    model_config: dict,
    *value_grad_args,
    **value_grad_kwargs
    ):

    organizer = su.StateOrganizer(opt_tree, opt_config)

    (model_tree, *value), grads = value_and_grad_fn(model_tree, model_config, *value_grad_args, **value_grad_kwargs)

    params, rest = organizer.params_filter(model_tree)

    updates, organizer.opt_state = organizer.opt_apply(grads, organizer.opt_state, params)

    params = optax.apply_updates(params, updates)

    model_tree = organizer.params_merger(rest, params)

    log_data = {}

    return organizer.get_state(), model_tree, log_data, *value
