

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


def decrement_jit_depth():
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

    FUNCTION_BUFFER.append(jax.tree_util.Partial(func))
    ARGS_BUFFER.append(args)
    KWARGS_BUFFER.append(kwargs)

    flush_calls()


def sidecall_wrapper(wrapper):

    @functools.wraps(wrapper)
    def wrapped_wrapper(func, *wrap_args, **wrap_kwargs):

        @functools.wraps(func)
        def func_returning_side_args(*args, **kwargs):
            global FUNCTION_BUFFER, ARGS_BUFFER, KWARGS_BUFFER
            output = func(*args, **kwargs)

            if not isinstance(output, tuple):
                output = [output]
            output = list(output)

            return tuple(output + [FUNCTION_BUFFER, ARGS_BUFFER, KWARGS_BUFFER])

        wrapped_func_returning_side_args = wrapper(func_returning_side_args, *wrap_args, **wrap_kwargs)

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            flush_calls()

            increment_jit_depth()

            values = wrapped_func_returning_side_args(*args, **kwargs)
            output = values[:-3]
            function_buffer, args_buffer, kwargs_buffer = tuple(values[-3:])

            decrement_jit_depth()

            if not is_in_jit():
                global FUNCTION_BUFFER, ARGS_BUFFER, KWARGS_BUFFER
                ARGS_BUFFER = args_buffer
                KWARGS_BUFFER = kwargs_buffer
                FUNCTION_BUFFER = function_buffer

            flush_calls()
            if len(output) == 1:
                return output[0]
            else:
                return tuple(output)

        return wrapped_func

    return wrapped_wrapper





