

import jax
from jax import numpy as jnp
import functools


FUNCTION_BUFFER = []
ARGS_BUFFER = []
KWARGS_BUFFER = []

JIT_DEPTH = 0



def increment_jit_depth():
    global JIT_DEPTH
    JIT_DEPTH = JIT_DEPTH + 1


def decement_jit_depth():
    global JIT_DEPTH
    assert JIT_DEPTH>0, "attempted to decrement JIT_DEPTH to a negative value!"

    JIT_DEPTH = JIT_DEPTH - 1


def is_in_jit():
    return JIT_DEPTH > 0


def flush_calls():
    global FUNCTION_BUFFER, ARGS_BUFFER, KWARGS_BUFFER
    if is_in_jit():
        return

    for func, args, kwargs in zip(FUNCTION_BUFFER, ARGS_BUFFER, KWARGS_BUFFER):
        func(*args, **kwargs)

    FUNCTION_BUFFER = []
    ARGS_BUFFER = []
    KWARGS_BUFFER = []


def sidecall(func, *args, **kwargs):
    global FUNCTION_BUFFER, ARGS_BUFFER, KWARGS_BUFFER

    FUNCTION_BUFFER.append(func)
    ARGS_BUFFER.append(args)
    KWARGS_BUFFER.append(kwargs)

    flush_calls()


def sidecall_wrapper(wrapper):
    global ARGS_BUFFER, KWARGS_BUFFER

    @functools.wraps(wrapper)
    def wrapped_wrapper(func, *wrap_args, **wrap_kwargs):

        @functools.wraps(func)
        def plumbed_func(*args, **kwargs):

            output = func(*args, **kwargs)

            return output, ARGS_BUFFER, KWARGS_BUFFER

        wrapped_plumbed_func = wrapper(plumbed_func, *wrap_args, **wrap_kwargs)

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            increment_jit_depth()

            output, args_buffer, kwargs_buffer = wrapped_plumbed_func(*args, **wrap_kwargs)
            ARGS_BUFFER = args_buffer
            KWARGS_BUFFER = kwargs_buffer

            decrement_jit_depth()

            flush_calls()

            return output

        return wrapped_func

    return wrapped_wrapper





