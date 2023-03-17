import jax
import numpy as np
from jax import numpy as jnp
from brachy import nn
from brachy import rng_util
from jax.tree_util import tree_map, tree_reduce

import einops

import pprint
import torch

import unittest

from brachy import structure_util as su

from brachy.optim.sgd import SGD
from brachy.optim.adamw import AdamW
from brachy.optim import mixed_precision_tree, mixed_precision_loss

from brachy.optim.process_grads import clip_grads
from brachy.optim.random_scaling import random_scale
from brachy import optim

try:
    import optax
except:
    optax = None




class T_FF(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.lin1 = torch.nn.Linear(4,10, bias=False)
        self.seq = torch.nn.Sequential(*[
            torch.nn.Conv2d(3,3,3,padding='same'),
            torch.nn.BatchNorm2d(3)
        ])
        # self.conv1 = torch.nn.Conv2d(3,3,3,padding='same')
        # self.bn1 = torch.nn.BatchNorm2d(3)
        self.lin2 = torch.nn.Linear(10,4)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.seq(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        x = self.lin2(x)
        return torch.sum(x)

@su.organized_init_with_rng
def simple_ff(organizer, rng=None):
    organizer.lin1 = nn.Linear(4,10, bias=False)
    
    organizer.seq = nn.Sequential(*[
        nn.Conv2d(3,3,3,padding='same'),
        nn.BatchNorm2d(3)
    ])
    organizer.lin2 = nn.Linear(10,4)
    organizer.set_apply(simple_ff_apply)


    organizer.seq[1].register_aux('force_high_precision', True)


@su.organized_apply
def simple_ff_apply(organizer, x):
    x = organizer.lin1(x)
    x = nn.functional.relu(x)
    x = organizer.seq(x)
    x = organizer.lin2(x)
    return jnp.sum(x)




def check_optimizes(opt_init):

        rng = jax.random.PRNGKey(0)

        tree, global_config = simple_ff(rng=rng)

        # state, apply = su.bind_module(tree, global_config)
        # tree = su.fill_tree_from_torch_module(tree, t_module)

        opt_tree, opt_config = opt_init(tree)

        def loss(tree, config, x):
            s, y = su.apply_tree(tree, config, x)
            return s, jnp.abs(y)

        def train_step(opt_tree, opt_config, tree, global_config, x):
            value_grad_fn = su.tree_value_and_grad(loss)
            # (state, value), grad = value_grad_fn(state, x)
            # l_t = lambda state: value_grad_fn(state, x)
            return su.apply_tree(opt_tree, opt_config, {}, value_grad_fn, tree, global_config, x)
    
        train_step = su.jit(train_step, static_argnums=(1,3))

        for i in range(100):

            x = jnp.ones(shape=(4,3,5,4))
            x_t = torch.tensor(np.array(x))
            
            opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config, x)


        assert jnp.allclose(value, 0.0, atol=1e-1), f"loss value not small after 100 iterations: value: {value}"



class TestSGD(unittest.TestCase):


    def test_simplest_sgd(self):
        t_module = torch.nn.Linear(4,1, bias=False)

        rng = jax.random.PRNGKey(0)

        model_tree, model_config = nn.Linear(4, 1, bias=False, rng=rng)
        
        with torch.no_grad():
            t_module.weight.mul_(0).add_(1)
        


        # state, apply = su.bind_module(tree, global_config)
        model_tree = su.fill_tree_from_torch_module(model_tree, t_module)
        model_tree, model_config = nn.functional.chain((model_tree, model_config), jnp.sum)

        opt_tree, opt_config = SGD(model_tree, lr=1.0, momentum=0.9, weight_decay=0.1)


        def train_step(opt_tree, opt_config, model_tree, model_config, x):
            value_grad_fn = su.tree_value_and_grad(model_tree['apply'])
            # (state, value), grad = value_grad_fn(state, x)
            # l_t = lambda state: value_grad_fn(state, x)
            hparams = {'lr': 0.001}
            return su.apply_tree(opt_tree, opt_config, hparams, value_grad_fn, model_tree, model_config, x)
        jit = su.improved_static(jax.jit)
        train_step = jit(train_step, static_argnums=(1,3))


        sgd_t = torch.optim.SGD(t_module.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)

        for i in range(100):

            x = jnp.ones(shape=(4)) * jnp.sqrt(i+1)
            x_t = torch.tensor(np.array(x))
            opt_tree, model_tree, logs, value = train_step(opt_tree, opt_config, model_tree, model_config, x)
            # tree = su.merge_trees(tree, tree_update)

            sgd_t.zero_grad()
            value_t = t_module(x_t)
            value_t.backward()
            sgd_t.step()

            value_t = value_t.detach().numpy()

            assert jnp.allclose(value, value_t), f"values not close on iteration {i}: jax value: {value}, torch value: {value_t}"





    def test_larger_sgd(self):
        t_module = T_FF()

        rng = jax.random.PRNGKey(0)

        tree, global_config = simple_ff(rng=rng)

        # state, apply = su.bind_module(tree, global_config)
        tree = su.fill_tree_from_torch_module(tree, t_module)

        opt_tree, opt_config = SGD(tree, lr=0.001, momentum=0.9, weight_decay=0.1)


        def train_step(opt_tree, opt_config, tree, global_config, x):
            value_grad_fn = su.tree_value_and_grad(tree['apply'])
            # (state, value), grad = value_grad_fn(state, x)
            # l_t = lambda state: value_grad_fn(state, x)
            return su.apply_tree(opt_tree, opt_config, {}, value_grad_fn, tree, global_config, x)
    
        train_step = su.jit(train_step, static_argnums=(1,3))


        sgd_t = torch.optim.SGD(t_module.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)

        for i in range(10):

            x = jnp.ones(shape=(4,3,5,4)) * jnp.sqrt(i+1)
            x_t = torch.tensor(np.array(x))
            
            opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config, x)

            sgd_t.zero_grad()
            value_t = t_module(x_t)
            value_t.backward()
            sgd_t.step()

            value_t = value_t.detach().numpy()

            assert jnp.allclose(value, value_t, rtol=1e-4), f"values not close on iteration {i}: jax value: {value}, torch value: {value_t}"



    def test_adamw(self):
        t_module = T_FF()

        rng = jax.random.PRNGKey(0)

        tree, global_config = simple_ff(rng=rng)

        tree = su.fill_tree_from_torch_module(tree, t_module)


        opt_tree, opt_config = AdamW(tree, lr=0.001)
        jit_canary = 0
        def train_step(opt_tree, opt_config, model_tree, global_config, x):
            nonlocal jit_canary
            jit_canary += 1
            value_grad_fn = su.tree_value_and_grad(model_tree['apply'])
            return su.apply_tree(opt_tree, opt_config, {}, value_grad_fn, model_tree, global_config, x)

        train_step = su.jit(train_step, static_argnums=(1,3))


        opt_t = torch.optim.AdamW(t_module.parameters(), lr=0.001)

        for i in range(10):

            x = jnp.ones(shape=(4,3,5,4)) * jnp.sqrt(i+1)
            x_t = torch.tensor(np.array(x))
            
            opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config, x)
            # tree = su.merge_trees(tree, update)

            opt_t.zero_grad()
            value_t = t_module(x_t)
            value_t.backward()
            opt_t.step()

            value_t = value_t.detach().numpy()


            assert jnp.allclose(value, value_t, rtol=1e-4), f"values not close on iteration {i}: jax value: {value}, torch value: {value_t}"
        

        assert jit_canary == 1


    def test_mixed_precision(self):

        rng = jax.random.PRNGKey(0)
        tree, global_config = simple_ff(rng=rng)

        mixed_tree, global_config = mixed_precision_tree((tree, global_config), loss_scalar=16.0)


        opt_tree, opt_config = SGD(tree, lr=0.0001, momentum=0.9, weight_decay=0.1)

        mixed_opt_tree, mixed_opt_config = SGD(mixed_tree, lr=0.0001, momentum=0.9, weight_decay=0.1)

        def loss(tree, global_config, x):
            tree, value = su.apply_tree(tree, global_config, x)
            return  tree,  0.1*value**2

        value_grad_fn = su.tree_value_and_grad(loss)

        mixed_value_grad_fn = su.tree_value_and_grad(mixed_precision_loss(loss))


        def train_step(opt_tree, opt_config, tree, global_config, x):
            return su.apply_tree(opt_tree, opt_config, {}, value_grad_fn, tree, global_config, x)

        def mixed_train_step(opt_tree, opt_config, tree, global_config, x):
            return su.apply_tree(opt_tree, opt_config, {}, mixed_value_grad_fn, tree, global_config, x)
    
        train_step = su.jit(train_step, static_argnums=(1,3))
        mixed_train_step = su.jit(mixed_train_step, static_argnums=(1,3))


        for i in range(100):

            x = jnp.ones(shape=(4,3,5,4)) * jnp.sqrt(i+1)
            x_t = torch.tensor(np.array(x))
            
            opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config, x)
            # tree = su.merge_trees(tree, tree_update)


            mixed_opt_tree, mixed_tree, logs, mixed_value = mixed_train_step(mixed_opt_tree, mixed_opt_config, mixed_tree, global_config, x)
            # mixed_tree = su.merge_trees(mixed_tree, mixed_tree_update)

            if i <= 5:
                assert jnp.allclose(value, mixed_value, rtol=1e-1), f"values not close on iteration {i}: float32 value: {value}, float16 value: {mixed_value}"

        assert jnp.allclose(0, mixed_value, atol=1e-4), f"mixed_precision did not optimize! loss was {mixed_value}"



    def test_clipped_grad_optimizes(self):

        rng = jax.random.PRNGKey(0)
        tree, global_config = simple_ff(rng=rng)


        opt_tree, opt_config = SGD(tree, lr=0.0001, momentum=0.9, weight_decay=0.1)
        opt_tree, opt_config = clip_grads((opt_tree, opt_config), 0.1)


        def loss(tree, global_config, x):
            tree, value = su.apply_tree(tree, global_config, x)
            return  tree,  0.1*value**2

        value_grad_fn = su.tree_value_and_grad(loss)



        def train_step(opt_tree, opt_config, tree, global_config, x):
            return su.apply_tree(opt_tree, opt_config, {}, value_grad_fn, tree, global_config, x)
    
        train_step = su.jit(train_step, static_argnums=(1,3))


        for i in range(100):

            x = jnp.ones(shape=(4,3,5,4)) * jnp.sqrt(i+1)
            x_t = torch.tensor(np.array(x))
            
            opt_tree, tree, logs, value = train_step(opt_tree, opt_config,  tree, global_config, x)
            # tree = su.merge_trees(tree, tree_update)

        assert jnp.allclose(0, value, atol=1e-4), f"clip grads did not optimize! loss was {value}"


    def test_clipped_grad_explicit(self):

        rng = jax.random.PRNGKey(0)

        tree = {
            'params': {
                'x': jnp.ones(2)
            },
            'buffers': {},
            'aux': {},
            'apply': lambda t, g, w: (t, t['params']['x'] * w),
            'submodules': {}
        }
        global_config = {}

        w = jnp.array([10.0, -1.0])


        opt_tree, opt_config = SGD(tree, lr=1.0, momentum=0.0, weight_decay=0.0)

        opt_tree, opt_config = clip_grads((opt_tree, opt_config), 1.0)



        def loss(tree, global_config, w):
            tree, value = su.apply_tree(tree, global_config, w)
            return  tree,  jnp.sum(value)

        value_grad_fn = su.tree_value_and_grad(loss)



        def train_step(opt_tree, opt_config, tree, global_config, w):
            return su.apply_tree(opt_tree, opt_config, {}, value_grad_fn, tree, global_config, w)

        train_step = su.jit(train_step, static_argnums=(1,3))

        opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config, w)

        new_x = tree['params']['x']

        assert jnp.allclose(new_x, jnp.array([0.0, 2.0])), f"x was unexpected value: {new_x}"



    def test_random_scale_offset(self):

        rng = jax.random.PRNGKey(0)

        tree = {
            'params': {
                'x': jnp.ones(2)
            },
            'buffers': {},
            'aux': {},
            'apply': lambda t, g, w: (t, 0.5*(t['params']['x'] * w)**2),
            'submodules': {}
        }
        global_config = {}

        w = jnp.array([6.0, -4.0])


        opt_tree, opt_config = SGD(tree, lr=1.0, momentum=0.0, weight_decay=0.0)

        opt_tree, opt_config = random_scale((opt_tree, opt_config), tree, distribution=lambda x: 0.75, rng=jax.random.PRNGKey(1))



        def loss(tree, global_config, w):
            tree, value = su.apply_tree(tree, global_config, w)
            return  tree,  jnp.sum(value)

        value_grad_fn = su.tree_value_and_grad(loss)



        def train_step(opt_tree, opt_config, tree, global_config, w):
            return su.apply_tree(opt_tree, opt_config, {}, value_grad_fn, tree, global_config, w)

        train_step = su.jit(train_step, static_argnums=(1,3))

        opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config, w)

        new_x = tree['params']['x']

        assert jnp.allclose(new_x, jnp.array([-26.0, -11.0])), f"x was unexpected value: {new_x}"

        opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config, w)

        new_x = tree['params']['x']

        assert jnp.allclose(new_x, jnp.array([667.0, 117.0])), f"x was unexpected value: {new_x}"



    def test_random_scale_independent_random(self):

        rng = jax.random.PRNGKey(0)

        x = jnp.ones(1)
        tree = {
            'params': {
                'x': x
            },
            'buffers': {},
            'aux': {},
            'apply': lambda t, g: (t, jnp.abs(t['params']['x'])),
            'submodules': {}
        }
        global_config = {}


        opt_tree, opt_config = SGD(tree, lr=1.0, momentum=0.0, weight_decay=0.0)

        opt_tree, opt_config = random_scale((opt_tree, opt_config), tree, rng=jax.random.PRNGKey(1))



        def loss(tree, global_config):
            tree, value = su.apply_tree(tree, global_config)
            return  tree,  jnp.sum(value)

        value_grad_fn = su.tree_value_and_grad(loss)



        def train_step(opt_tree, opt_config, tree, global_config):
            return su.apply_tree(opt_tree, opt_config, {}, value_grad_fn, tree, global_config)

        train_step = su.jit(train_step, static_argnums=(1,3))

        opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config)

        new_x = tree['params']['x']

        offset = new_x - x
        
        true_params = opt_tree['buffers']['true_params']['params']['x']
        assert jnp.allclose(true_params, jnp.array([0.0])), f"x was unexpected value: {true_params}"

        opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config)

        offset2 = tree['params']['x'] - new_x

        true_params = opt_tree['buffers']['true_params']['params']['x']
        assert jnp.allclose(true_params, jnp.array([-1.0])), f"x was unexpected value: {true_params}"
        
        assert not jnp.allclose(offset, offset2)




    def test_random_scale_no_interpolate(self):

        rng = jax.random.PRNGKey(0)

        tree = {
            'params': {
                'x': jnp.ones(2)
            },
            'buffers': {},
            'aux': {},
            'apply': lambda t, g, w: (t, 0.5*(t['params']['x'] * w)**2),
            'submodules': {}
        }
        global_config = {}

        w = jnp.array([6.0, -4.0])


        opt_tree, opt_config = SGD(tree, lr=1.0, momentum=0.0, weight_decay=0.0)

        opt_tree, opt_config = random_scale((opt_tree, opt_config), tree, distribution=lambda x: 0.75, interpolate=False, rng=jax.random.PRNGKey(1))



        def loss(tree, global_config, w):
            tree, value = su.apply_tree(tree, global_config, w)
            return  tree,  jnp.sum(value)

        value_grad_fn = su.tree_value_and_grad(loss)



        def train_step(opt_tree, opt_config, tree, global_config, w):
            return su.apply_tree(opt_tree, opt_config, {}, value_grad_fn, tree, global_config, w)

        train_step = su.jit(train_step, static_argnums=(1,3))

        opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config, w)

        new_x = tree['params']['x']

        assert jnp.allclose(new_x, jnp.array([-26.0, -11.0])), f"x was unexpected value: {new_x}"

        opt_tree, tree, logs, value = train_step(opt_tree, opt_config, tree, global_config, w)

        new_x = tree['params']['x']

        assert jnp.allclose(new_x, jnp.array([676.0, 121.0])), f"x was unexpected value: {new_x}"


    def test_optax(self):
        if optax is None:
            return

        optax_wrap  = lambda *a, **kw: optim.optax.init(optax.adam(learning_rate=1e-4), *a, **kw)

        check_optimizes(optax_wrap)