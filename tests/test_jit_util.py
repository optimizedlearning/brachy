import jax
import numpy as np
from jax import numpy as jnp
from brachy import nn
from brachy import rng_util

from tempfile import TemporaryFile
# from jax.tree_util import tree_map, tree_reduce

from jax._src.typing import Array, ArrayLike, DType, DTypeLike
from typing import overload, Any, Callable, Literal, Optional, Sequence, Tuple, Union

import einops

import pprint
import torch

from brachy import structure_util as su
from brachy import jit_util


import unittest

def same_dicts(*to_compare, keys_to_exclude=[]):
    if len(to_compare) == 0:
        return True
    
    keys = set(to_compare[0].keys())

    for d in to_compare[1:]:
        if set(d.keys()) != keys:
            return False

    for key in keys:
        if key in keys_to_exclude:
            continue
        value = to_compare[0][key]

        for d in to_compare[1:]:
            comp_value = d[key]

            if type(value) != type(comp_value):
                return False

            if isinstance(value, Array) or isinstance(value, np.ndarray):
                if value.shape != comp_value.shape:
                    return False
                if not jnp.allclose(value, comp_value):
                    return False
                continue

            if not isinstance(value, dict):
                if value != comp_value:
                    return False
                continue
            
            if not same_dicts(value, comp_value, keys_to_exclude=keys_to_exclude):
                return False

    
    return True



def same_trees(*trees, keys_to_exclude=[]):
    for tree in trees:
        if not su.is_structure_tree(tree, recurse=True):
            return False

    return same_dicts(*trees,keys_to_exclude=keys_to_exclude)





def apply(tree, global_config, x, y):
    value = tree['params']['a'] + x -y
    split = tree.filter_keys(['params', 'buffers'])
    return tree, value

def apply_child(tree, global_config, x, y):
    value = 3*tree['params']['f'] + tree['buffers']['g'] + x -y
    split = tree.filter_keys(['params', 'buffers'])
    return tree, value

def apply_grand_child(tree, global_config, x, y):
    value = -1*tree['params']['f'] + tree['buffers']['g'] + x -y
    split = tree.filter_keys(['params', 'buffers'])
    return tree, value

grand_child = {
    'params': {
        'f': jnp.zeros((2,3)),
    },
    'buffers': {
        'g': jnp.array([[33,99,3],[5,6,7]])
    },
    'static': {
        'comment': 'this is a grand child node',
    },
    'apply': apply_grand_child,
    'submodules': {}
}


child = {
    'params': {
        'f': jnp.zeros((1,3)),
    },
    'buffers': {
        'g': jnp.array([[1,99,3],[5,6,7]])
    },
    'static': {
        'comment': 'this is a child node',
    },
    'apply': apply_child,
    'submodules': {
        'g': grand_child
    }
}

tree = {
    'params': {
        'a': jnp.ones(4),
        'b': jnp.zeros((1,2,1)),
    },
    'buffers': {
        'x': jnp.array([[1,2,3],[5,6,7]])
    },
    'static': {
        'comment': 'this is some text',
        4: 234
    },
    'apply': apply,
    'submodules': {
        'c': child
    }
}

params = {
    'params': {
        'a': jnp.ones(4),
        'b': jnp.zeros((1,2,1)),
    },
    'submodules': {
        'c': {
            'params': {
                'f': jnp.zeros((1,3))
            },
            'submodules': {
                'g': {
                    'params': {
                        'f': jnp.zeros((2,3)),
                    },
                    'submodules': {}
                }
            }
        }
    }
}


buffers = {
    'buffers': {
        'x': jnp.array([[1,2,3],[5,6,7]])
    },
    'submodules': {
        'c': {
            'buffers': {
                'g': jnp.array([[1,99,3],[5,6,7]])
            },
            'submodules': {
                'g': {
                    'buffers': {
                        'g': jnp.array([[33,99,3],[5,6,7]])
                    },
                    'submodules': {}
                }
            }
        }
    }
}

apply_static = {
    'apply': apply,
    'static': {
        'comment': 'this is some text',
        4: 234
    },
    'submodules': {
        'c': {
            'apply': apply_child,
            'static': {
                'comment': 'this is a child node',
            },
            'submodules': {
                'g': {
                    'apply': apply_grand_child,
                    'static': {
                        'comment': 'this is a grand child node',
                    },
                    'submodules': {}
                }
            }
        }
    }
}


def grandchild_module():
    global_config = {
        'test_override': 'grandchild',
        'test_override_grandchild': 'g',
        'unique_grandchild': 'g',
    }


    params = {
        'w': jnp.array([1,2,3,4,5])
    }
    buffers = {
        'g': jnp.array([-1,-1,-1,-1,3])
    }
    static = {
        'description': 'grandchild module'
    }

    tree = {
        'params': params,
        'buffers': buffers,
        'static': static,
        'apply': grandchild_apply,
        'submodules': {}
    }

    return tree, global_config

def grandchild_apply(tree, global_config, x):
    w = tree['params']['w']
    g = tree['buffers']['g']

    y = x*w + g

    assert global_config['test_override'] == 'root'
    assert global_config['unique_child'] == 'c'

    return su.filter_keys(tree), y


def child_module(p):
    global_config = {
        'test_override': 'child',
        'unique_child': 'c'
    }

    organizer = su.StateOrganizer(global_config=global_config)

    organizer.update_global_config({'test_override_grandchild': 'c'})

    organizer.gc = grandchild_module()

    organizer.w = jnp.array([p])

    return organizer.create_module(child_apply)


def child_apply(tree, global_config, x):

    organizer = su.StateOrganizer(tree, global_config)

    assert organizer.get_global_config('unique_grandchild') == 'g'
    assert organizer.get_global_config('test_override') == 'root'

    y = organizer.gc(x)

    y = y * organizer.w

    return organizer.get_state(), y


def root_module():

    organizer = su.StateOrganizer()
    organizer.update_global_config('test_override', 'root')
    organizer.register_buffer('a', jnp.array([1,1,1,3,3]))

    for k in range(1,4):
        organizer.register_submodule(k, child_module(k))

    return organizer.create_module(root_apply)

def root_apply(tree, global_config, x):

    organizer = su.StateOrganizer(tree, global_config)

    assert organizer.get_global_config('test_override_grandchild') ==  'c'

    y1 = organizer[1](x)
    y2 = organizer[2](y1)
    y3 = organizer[3](x)

    y_final = y1+y2+y3 + organizer.a

    organizer.a = jnp.array([1,1,1,2,2])

    return organizer.get_state(), y_final

def alt_root_apply(tree, global_config, x):

    organizer = su.StateOrganizer(tree, global_config)

    assert organizer.get_global_config('test_override_grandchild') ==  'c'

    y1 = organizer[1](x)
    y2 = organizer[2](y1)
    y3 = organizer[3](x)

    y_final = y1+y2+y3 + organizer.a

    return organizer.get_state(), y_final

    

def assert_close(a, b):
    assert jnp.allclose(a, b), f"{a} is not close to {b}!"

class TestStructureUtils(unittest.TestCase):









    def test_jit_autostatic(self):


        trace_count = 0
        jit  = jit_util.improved_static(jax.jit)

        @jit
        def foo(x,y):
            nonlocal trace_count
            trace_count += 1
            if x['q'] == 'go ahead!':
                return {'a': x['a'], 'b': y['b']}
            else:
                return {'a': 2*y['a'], 'b': y['b']}

        x = {
            'q': 'stop',
            'a': jnp.ones(3)
        }
        y = {
            'a': jnp.ones(5),
            'b': ['hello', 'friend']
        }

        z = foo(x,y)
        x['a'] = jnp.zeros(3)
        w = foo(x,y)

        assert jnp.allclose(z['a'], jnp.array([2.0,2.0,2.0,2.0,2.0]))
        assert z['b'][0] == 'hello'
        assert z['b'][1] == 'friend'
        assert trace_count == 1

    def test_jit_tree(self):

        trace_count = 0

        @jit_util.improved_static(jax.jit)
        def func(tree, global_config, x):
            nonlocal trace_count
            trace_count += 1
            organizer = su.StateOrganizer(tree, global_config)
            y = jnp.matmul(x, organizer.weight)+ organizer.bias
            return organizer.get_state(), y

        lin, global_config = nn.Linear(5,5, rng=jax.random.PRNGKey(0))
        global_config['foo'] = 'hihihi'
        lin['params']['weight'] = jnp.eye(5)
        lin['params']['bias'] = jnp.ones(5)
        x = jnp.ones(5)


        lin, y = func(lin, global_config, x)
        lin, y = func(lin, global_config, x)

        assert jnp.allclose(y, 2*jnp.ones(5)), f"y was: {y}"
        assert trace_count == 1, f"trace count was: {trace_count}"

        def loss(tree, global_config, x):
            state, y = func(tree, global_config, x)
            return state, jnp.sum(y**2)

        loss = jit_util.jit(loss, static_argnums=1)


        value_and_grad = su.tree_value_and_grad(loss)
        (lin, value), grad = value_and_grad(lin, global_config, x)

        assert jnp.allclose(value, 20)
        assert jnp.allclose(grad['params']['bias'], 2*2*jnp.ones(5)), f"bias: {grad['params']['bias']}"
        assert jnp.allclose(grad['params']['weight'], 2*2*jnp.ones((5,5))), f"bias: {grad['params']['bias']}"


        def non_jittable_loss(tree, global_config, x):
            organizer = su.StateOrganizer(tree, global_config)
            y = jnp.matmul(x, organizer.weight)+ organizer.bias
            return organizer.get_state(), jnp.sum(y**2)

        value_and_grad = su.tree_value_and_grad(non_jittable_loss)
        (update, value), grad = value_and_grad(lin, global_config, x)

        assert jnp.allclose(value, 20)
        assert jnp.allclose(grad['params']['bias'], 2*2*jnp.ones(5)), f"bias: {grad['params']['bias']}"
        assert jnp.allclose(grad['params']['weight'], 2*2*jnp.ones((5,5))), f"bias: {grad['params']['bias']}"

    def test_jit_nested_tree(self):


        trace_count = 0

        @jit_util.improved_static(jax.jit)
        def func(nested_tree, global_config, x):
            nonlocal trace_count
            tree = nested_tree['nested']
            trace_count += 1
            organizer = su.StateOrganizer(tree, global_config)
            y = jnp.matmul(x, organizer.weight)+ organizer.bias
            return {'nested': organizer.get_state()}, y

        lin, global_config = nn.Linear(5,5, rng=jax.random.PRNGKey(0))
        lin['static'] = {'random': 'value'}
        global_config['foo'] = 'hihihi'
        lin['params']['weight'] = jnp.eye(5)
        lin['params']['bias'] = jnp.ones(5)
        x = jnp.ones(5)


        lin, y = func({'nested': lin}, global_config, x)
        lin = lin['nested']
        lin, y = func({'nested': lin}, global_config, x)
        lin = lin['nested']

        assert jnp.allclose(y, 2*jnp.ones(5)), f"y was: {y}"
        assert trace_count == 1, f"trace count was: {trace_count}"

    def test_jit_static_return(self):

        trace_count = 0

        def func(tree, global_config, x, z):
            nonlocal trace_count
            trace_count += 1
            organizer = su.StateOrganizer(tree, global_config)
            y = jnp.matmul(x, organizer.weight)+ organizer.bias
            return organizer.get_state(), global_config, z

        j_func = jit_util.jit(func, static_argnums=3, static_returns=2)

        lin, global_config = nn.Linear(5,5, rng=jax.random.PRNGKey(0))
        lin['static'] = {
            'number': 10
        }
        global_config['foo'] = 'hihihi'
        lin['params']['weight'] = jnp.eye(5)
        lin['params']['bias'] = jnp.ones(5)
        x = jnp.ones(5)
        z= {'f': 0, 'g': False}

        state, y, z = j_func(lin, global_config, x, z)
        # state, y, z = j_func(lin, global_config, x, z)

        assert state['static']['number'] == 10
        assert not isinstance(state['static']['number'], Array)

        assert z['g'] == False
        assert z['f'] == 0
        assert not isinstance(z['f'], Array)




    def test_jit_notree(self):

        trace_count = 0

        def func(x, y, z, w, q):
            nonlocal trace_count
            trace_count += 1
            if z['x']['y']:
                return x + w[3] + q
            else:
                if y['a']:
                    return -x - w[1] - q
                else:
                    return x
        jit = jit_util.improved_static(jax.jit)
        j_func = jit(func, static_argnums=1, static_argnames=('z','w'))

        x = jnp.ones(1)
        other_x = jnp.zeros(1)
        q = jnp.ones(1)
        other_q = jnp.zeros(1)
        y = {
            'a': True
        }
        z = {
            'x': {
                'y': True
            }
        }
        w = [1,2,3,4]

        a = j_func(x, y, z, w, q)
        assert jnp.allclose(a, 6), f"a value: {a}"
        assert trace_count == 1, f"trace count: {trace_count}"

        a = j_func(other_x, y, z, w, other_q)
        assert jnp.allclose(a, 4), f"a value: {a}"
        assert trace_count == 1, f"trace count: {trace_count}"

        z['x']['y'] = False

        a = j_func(other_x, y, z, w, other_q)
        assert jnp.allclose(a, -2), f"a value: {a}"
        assert trace_count == 2, f"trace count: {trace_count}"

        other_y = {
            'a': True
        }

        other_z ={
            'x': {
                'y': False
            }
        }

        a = j_func(other_x, other_y, other_z, w, other_q)
        assert jnp.allclose(a, -2), f"a value: {a}"
        assert trace_count == 2, f"trace count: {trace_count}"

        other_z['p'] = 5
        a = j_func(other_x, other_y, other_z, w, other_q)
        assert jnp.allclose(a, -2), f"a value: {a}"
        assert trace_count == 3, f"trace count: {trace_count}"

        a = j_func(other_x, other_y, z=other_z, w=w, q=other_q)
        assert jnp.allclose(a, -2), f"a value: {a}"
        assert trace_count == 4, f"trace count: {trace_count}"

        other_y['a'] = False
        a = j_func(other_x, other_y, z=other_z, w=w, q=other_q)
        assert jnp.allclose(a, 0), f"a value: {a}"
        assert trace_count == 5, f"trace count: {trace_count}"


        other_y = {
            'a': False
        }
        a = j_func(x, other_y, z=other_z, w=w, q=other_q)
        assert jnp.allclose(a, 1), f"a value: {a}"
        assert trace_count == 5, f"trace count: {trace_count}"


    def test_jit_side_channel(self):

        log_dict = {}

        def add_to_logs(value, key):
            log_dict[value] =  key


        def func(x):
            
            jit_util.sidecall(add_to_logs, 'test2', x+2)
            jit_util.sidecall(add_to_logs, 'test1', x+1)
            jit_util.sidecall(add_to_logs, 'test2', x-1)

            return 2*x

        j_func = jit_util.jit(func)

        z = j_func(6)
        z2 = j_func(5)

        assert jnp.allclose(z, 12), f"returned in correct value: {z}"
        assert jnp.allclose(z2, 10), f"returned in correct value: {z2}"
        assert_close(log_dict['test1'], 6)
        assert_close(log_dict['test2'], 4) 


    def test_jit_side_channel_nested(self):

        log_dict = {'count': 0}

        def add_to_logs(value, key):
            if value in log_dict and log_dict[value] > 50:
                log_dict['count'] += 1
            log_dict[value] =  key


        def func(x):
            
            jit_util.sidecall(add_to_logs, 'test2', 100)
            jit_util.sidecall(add_to_logs, 'test1', x+1)
            jit_util.sidecall(add_to_logs, 'test2', x-1)

            return 2*x


        def base_func(x):
            
            jit_util.sidecall(add_to_logs, 'test3', x+4)

            return func(x)

        j_func = jit_util.jit(base_func)

        z = j_func(6)
        z2 = j_func(5)

        assert jnp.allclose(z, 12), f"returned in correct value: {z}"
        assert jnp.allclose(z2, 10), f"returned in correct value: {z2}"
        assert_close(log_dict['test1'], 6)
        assert_close(log_dict['test2'], 4) 
        assert_close(log_dict['test3'], 9)
        assert_close(log_dict['count'], 2)