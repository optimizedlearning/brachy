import jax
import numpy as np
from jax import numpy as jnp
import nn
import rng_util
from jax.tree_util import tree_map, tree_reduce

import einops

import pprint
import torch

import unittest

import structure_util as su

from optim.sgd import SGD
from optim.adamw import AdamW


class T_FF(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.lin1 = torch.nn.Linear(4,10, bias=False)
        self.conv1 = torch.nn.Conv2d(3,3,3,padding='same')
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.lin2 = torch.nn.Linear(10,4)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lin2(x)
        return torch.sum(x)

@su.organized_init_with_rng
def simple_ff(organizer, rng=None):
    organizer.lin1 = nn.Linear(4,10, bias=False)
    organizer.bn1 = nn.BatchNorm2d(3)
    organizer.conv1 = nn.Conv2d(3,3,3,padding='same')
    organizer.lin2 = nn.Linear(10,4)
    organizer.set_apply(simple_ff_apply)

@su.organized_apply
def simple_ff_apply(organizer, x):
    x = organizer.lin1(x)
    x = nn.functional.relu(x)
    x = organizer.conv1(x)
    x = organizer.bn1(x)
    x = organizer.lin2(x)
    return jnp.sum(x)



class TestSGD(unittest.TestCase):


    def test_sgd(self):
        t_module = T_FF()

        rng = jax.random.PRNGKey(0)

        tree, global_config = simple_ff(rng=rng)

        state, apply = su.bind_module(tree, global_config)
        state = su.fill_tree_from_torch_module(state, t_module)

        sgd_state, sgd_apply = SGD(state, lr=0.001, momentum=0.9, weight_decay=0.1)

        @jax.jit
        def train_step(sgd_state, state, x):
            value_grad_fn = su.state_value_and_grad(apply)
            (state, value), grad = value_grad_fn(state, x)
            l_t = lambda state: value_grad_fn(state, x)
            return sgd_apply(sgd_state, state, l_t, lr=1.0)


        sgd_t = torch.optim.SGD(t_module.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)

        for i in range(10):

            x = jnp.ones(shape=(4,3,5,4)) * jnp.sqrt(i+1)
            x_t = torch.tensor(np.array(x))
            
            sgd_state, state, value = train_step(sgd_state, state, x)

            sgd_t.zero_grad()
            value_t = t_module(x_t)
            value_t.backward()
            sgd_t.step()

            value_t = value_t.detach().numpy()

            assert jnp.allclose(value, value_t), f"values not close on iteration {i}: jax value: {value}, torch value: {value_t}"



    def test_adamw(self):
        t_module = T_FF()

        rng = jax.random.PRNGKey(0)

        tree, global_config = simple_ff(rng=rng)

        state, apply = su.bind_module(tree, global_config)
        state = su.fill_tree_from_torch_module(state, t_module)

        opt_state, opt_apply = AdamW(state, lr=0.001)

        @jax.jit
        def train_step(opt_state, state, x):
            value_grad_fn = su.state_value_and_grad(apply)
            (state, value), grad = value_grad_fn(state, x)
            l_t = lambda state: value_grad_fn(state, x)
            return opt_apply(opt_state, state, l_t, lr=1.0)


        opt_t = torch.optim.AdamW(t_module.parameters(), lr=0.001)

        for i in range(10):

            x = jnp.ones(shape=(4,3,5,4)) * jnp.sqrt(i+1)
            x_t = torch.tensor(np.array(x))
            
            opt_state, state, value = train_step(opt_state, state, x)

            opt_t.zero_grad()
            value_t = t_module(x_t)
            value_t.backward()
            opt_t.step()

            value_t = value_t.detach().numpy()

            assert jnp.allclose(value, value_t), f"values not close on iteration {i}: jax value: {value}, torch value: {value_t}"




