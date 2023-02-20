import jax
import numpy as np
from jax import numpy as jnp
import nn
import rng_util
from jax.tree_util import tree_map, tree_reduce


import torch

import unittest


def allclose(a, b):
    return jnp.allclose(a, b)#, atol=1e-4, rtol=1e-4)

def t_to_np(tensor):
    return tensor.detach().numpy()


def tree_sub(a, b):
    return tree_map(lambda x, y: x-y, a, b)

def tree_sub_accum(a, b, c):
    return tree_map(lambda x, y, z: x+ y-z, a, b, c)

def tree_square_sub_accum(a, b, c):
    return tree_map(lambda x, y, z: x+ y**2 - z**2, a, b, c)


def tree_square_accum(a, b):
    return tree_map(lambda x, y: x+ y**2, a, b)


def tree_scale(a, c):
    return tree_map(lambda x: c*x, a)

# def tree_norm(a):
#     return jnp.sqrt(tree_reduce(lambda x, y: x + jnp.sum(y**2), a, 0.0))

# def tree_norm(a):
#     return jnp.abs(tree_reduce(lambda x, y: x + jnp.sum(y), a, 0.0))

def tree_norm(a):
    return tree_map(lambda x: jnp.abs(jnp.sum(x)), a)

def tree_sqrt(a):
    return tree_map(lambda x: jnp.sqrt(x), a)

def tree_div(a, b):
    return tree_map(lambda x,y: x/(y +1e-8), a, b)

def tree_reduce_max(a):
    return tree_reduce(lambda x, y: jnp.maximum(x,y), a, 0.0)

def tree_min(a, b):
    return tree_map(lambda x, y: jnp.minimum(x,y), a, b)


def tree_small(a, tol=1e-4):
    tree_norm(a) < tol

def zeros_like(a):
    return tree_map(lambda x: jnp.zeros_like(x), a)

def tree_size(a):
    return tree_reduce(lambda x, y: x+ y.size, a, 0)

class MyModule:
    def __new__(cls, vocab, embed, dim1, dim2, dim3=1, rng=None):
        if rng is None:
            rng = rng_util.split(1)

        organizer = nn.StateOrganizer()

        with rng_util.RNGState(rng):

            organizer.embed = nn.Embedding(vocab, embed)#embed)
            organizer.seq = nn.Sequential(nn.Linear(embed, dim1), nn.Linear(dim1, dim2))

            r = rng_util.split(1)

            # t_mul = torch.normal(torch.ones(dim2))
            mul = 1 + jax.random.normal(r, (dim2,))#t_mul.numpy()
            organizer.register_buffer('mul', mul)#, t_mul)

            organizer.fc2 = nn.Linear(dim2, dim3)

        return organizer.create_module(cls)#, MyTModule(organizer), return_torch)



    def apply(pack_state, state, x):

        module = pack_state(state)

        x = module.embed(x)
        x = module.seq(x)
        x = module.mul * x
        x = module.fc2(x)

        return x

class T_MyModule(torch.nn.Module):

    def __init__(self,vocab, embed, dim1, dim2, dim3=1):
        self.embed = torch.nn.Embedding(vocab, embed)
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(embed, dim1),
            torch.nn.Linear(dim1, dim2)
        )
        mul = torch.normal(torch.ones(dim2))
        self.register_buffer('mul', mul)
        self.fc2 = torch.nn.Linear(dim2, dim3)

    def forward(self, x):
        x = self.embed(x)
        x = self.seq(x)
        x = self.mul * x
        x = self.fc2(x)
        return x




class NextModule:

    def __new__(cls, vocab, embed, dim_next, dim_out, dim1, dim2, rng=None):
        if rng is None:
            rng = rng_util.split(1)

        organizer = nn.StateOrganizer()


        with rng_util.RNGState(rng):
            blah = MyModule(vocab, embed, dim1, dim2, dim_next)
            organizer.trunk = blah#MyModule(vocab, embed, dim1, dim2, dim_next)#, return_torch=return_torch)


            r = rng_util.split(1)

            bias = jax.random.normal(r, (dim_next,))

            # t_bias = torch.normal(torch.zeros(dim_next))
            # t_bias.requires_grad = True
            # j_bias = t_bias.detach().numpy()
            
            organizer.register_parameter('next_bias', bias)#j_bias, t_bias)

            organizer.head = nn.Linear(dim_next, dim_out)#, return_torch=return_torch)

        return organizer.create_module(cls)#, NextTModule(organizer), return_torch)



    def apply(module, state, x):
        module = module(state)


        x = module.trunk(x)
        x = jax.nn.relu(x)
        x = module.next_bias + x
        x = jax.nn.relu(x)
        x = module.head(x)

        return x


class T_NextModule(torch.nn.Module):

    def __init__(self, vocab, embed, dim_next, dim_out, dim1, dim2):
        self.trunk = T_MyModule(vocab, embed, dim1, dim2, dim_next)
        bias = torch.normal(torch.zeros(dim_next))
        self.register_parameter('next_bias', bias)
        self.head = torch.nn.Linear(dim_next, dim_out)

    def forward(self, x):
        x = self.trunk(x)
        x = torch.nn.functional.relu(x)
        x = self.next_bias + x
        x = torch.nn.functional.relu(x)
        x = self.head(x)

        return x


# def get_nested_state(t_module):
#     state = {
#         'params': {},
#         'constants': {}
#     }

#     params = {}
#     constants = {}

#     params['next_bias'] = t_to_np(t_module.next_bias)
#     params['head'] = {
#         'weight': t_to_np(t_module.head.weight),
#         'bias': t_to_np(t_module.head.bias)
#     }

#     constants['head'] = {}

    
#     embed_params = 

def test_initialization(rng, module_gen, t_module_gen, get_t_state, sample_num=1000):
    mean = None
    var = None

    base = None
    base_t = None
    for _ in range(sample_num):
        rng, subkey = jax.random.split(rng)
        state, apply = module_gen(subkey)
        t_module = t_module_gen()
        t_state  = get_t_state(t_module)

        if mean is None:
            mean = zeros_like(state)
            var = zeros_like(state)
            base = zeros_like(state)
            base_t = zeros_like(state)


        mean = tree_sub_accum(mean, state, t_state)
        var = tree_square_sub_accum(var, state, t_state)
        base = tree_square_accum(base, state)
        base_t = tree_square_accum(base, t_state)


    mean = tree_norm(tree_scale(mean, 1.0/(sample_num * tree_size(mean))))
    var = tree_norm(tree_scale(var, 1.0/(sample_num * tree_size(var))))
    base = tree_norm(tree_scale(base, 1.0/(sample_num * tree_size(base))))
    base_t = tree_norm(tree_scale(base_t, 1.0/(sample_num * tree_size(base_t))))

    min_base = tree_min(base, base_t)

    std = tree_sqrt(var)

    # print(f"base: {base}")
    # print(f"base_t: {base_t}")
    # print(f"var: {var}")
    assert tree_reduce_max(tree_div(mean,tree_sqrt(min_base))) < 2e-2, f"mean was too big:\nmean:\n{mean}\ndiv:\n{tree_div(mean,tree_sqrt(min_base))}"
    assert tree_reduce_max(tree_div(var, min_base)) < 2e-2, f"var was too big:\nvar:\n{var}\ndiv:\n{tree_div(var,min_base)}"#, {jnp.abs(base-base_t)/(base+base_t)}"



            


class TestNN(unittest.TestCase):



    # def test_identity(self):
    #     state, apply = nn.Identity(None)

    #     x = jnp.array([[1,2,3],[5,6,6],[7,8,9]], dtype=float)

    #     t_module = torch.nn.Identity()

    #     x_t = torch.tensor(np.array(x))

    #     y, _ = apply(state, x)

    #     y_t = t_module(x_t).numpy()

    #     self.assertTrue(allclose(y_t, y))



    # def test_linear(self):

    #     rng = jax.random.PRNGKey(0)

    #     def get_t_state(t_module):
    #         return {
    #             'params': {
    #                 'weight': t_to_np(t_module.weight),
    #                 'bias': t_to_np(t_module.bias)
    #             },
    #             'constants': {}
    #         }


    #     module_gen = lambda r: nn.Linear(300, 4000, bias=True, rng=r)
    #     t_module_gen = lambda : torch.nn.Linear(300, 4000, bias=True)
    #     test_initialization(rng, module_gen, t_module_gen, get_t_state, 100)



    #     _, apply = nn.Linear(3, 2, bias=True, rng=rng)
    #     t_module = torch.nn.Linear(3, 2, bias=True)
    #     state = get_t_state(t_module)

    #     x = jnp.array([[1,2,3],[5,6,6],[7,8,9]], dtype=jnp.float32)
    #     x_t = torch.tensor(np.array(x))


    #     y, state = apply(state, x)
    #     y2, _ = apply(state, x)
    #     y_t = t_module(x_t).detach().numpy()

    #     self.assertTrue(allclose(y_t, y))
    #     self.assertTrue(allclose(y_t, y2))


    # def test_conv2d(self):
    #     rng = jax.random.PRNGKey(0)

    #     def get_t_state(t_module):
    #         return {
    #             'params': {
    #                 'weight': t_to_np(t_module.weight),
    #                 'bias': t_to_np(t_module.bias)
    #             },
    #             'constants': {}
    #         }

    #     module_gen = lambda r: nn.Conv2d(30, 40, 50, padding='same', bias=True, rng=r)
    #     t_module_gen = lambda: torch.nn.Conv2d(30, 40, 50, padding='same', bias=True)
    #     test_initialization(rng, module_gen, t_module_gen, get_t_state, 100)


    #     _, apply = nn.Conv2d(3, 4, 5, padding='same', bias=True, rng=rng)
    #     t_module = torch.nn.Conv2d(3, 4, 5, padding='same', bias=True)
    #     state = get_t_state(t_module)

    #     x = jnp.array(np.random.normal(np.ones((2, 3, 6,7))), dtype=jnp.float32)
    #     x_t = torch.tensor(np.array(x))

    #     y, state = apply(state, x)
    #     y2, _ = apply(state, x)
    #     y_t = t_module(x_t).detach().numpy()

    #     self.assertTrue(allclose(y_t, y))
    #     self.assertTrue(allclose(y_t, y2))



    # def test_embedding(self):
    #     rng =  jax.random.PRNGKey(0)

    #     def get_t_state(t_module):
    #         return {
    #             'params': {
    #                 'weight': t_to_np(t_module.weight)
    #             },
    #             'constants': {}
    #         }

    #     module_gen = lambda r: nn.Embedding(500, 1000, rng=r)
    #     t_module_gen = lambda : torch.nn.Embedding(500, 1000)
    #     test_initialization(rng, module_gen, t_module_gen, get_t_state, 100)

    #     _, apply = nn.Embedding(30, 10, rng=rng)
    #     t_module = torch.nn.Embedding(30, 10)
    #     state = get_t_state(t_module)

    #     x = jnp.array([0, 2, 29, 7, 4])
    #     x_t = torch.tensor(np.array(x))

    #     y, state = apply(state, x)
    #     y2, _ = apply(state, x)
    #     y_t = t_module(x_t).detach().numpy()

    #     self.assertTrue(allclose(y_t, y))
    #     self.assertTrue(allclose(y_t, y2))

    # def test_sequential(self):
    #     rng =  jax.random.PRNGKey(0)
    #     def get_t_state(t_module):
    #         params = []
    #         constants = []
    #         for l in t_module:
    #             params.append({
    #                 'weight': t_to_np(l.weight),
    #                 'bias': t_to_np(l.bias)
    #             })
    #             constants.append({})
    #         state = {
    #             'params': params,
    #             'constants': constants
    #         }
    #         return state
            
    #     def module_gen(r):
    #         with rng_util.RNGState(r):
    #             chain = [
    #                 nn.Linear(3, 1000),
    #                 nn.Linear(1000, 500),
    #                 nn.Linear(500, 50)
    #             ]
    #             state, apply = nn.Sequential(*chain)
    #         return state, apply

    #     def t_module_gen():
    #         return torch.nn.Sequential(*[
    #             torch.nn.Linear(3, 1000),
    #             torch.nn.Linear(1000, 500),
    #             torch.nn.Linear(500, 50)
    #         ])

    #     test_initialization(rng, module_gen, t_module_gen, get_t_state, 500)

    #     with rng_util.RNGState(rng):
    #         chain = [
    #             nn.Linear(3, 10),
    #             nn.Linear(10, 20),
    #             nn.Linear(20, 3)
    #         ]
    #         _, apply = nn.Sequential(*chain)
    #     t_module = torch.nn.Sequential(*[
    #         torch.nn.Linear(3, 10),
    #         torch.nn.Linear(10, 20),
    #         torch.nn.Linear(20, 3)
    #     ])
    #     state = get_t_state(t_module)

    #     x = jnp.array([[1,2,3],[5,6,6],[7,8,9]], dtype=float)
    #     x_t = torch.tensor(np.array(x))

    #     y, state = apply(state, x)
    #     y2, _ = apply(state, x)
    #     y_t = t_module(x_t).detach().numpy()

    #     self.assertTrue(allclose(y_t, y))
    #     self.assertTrue(allclose(y_t, y2))


        
    # def test_layer_norm(self):
    #     rng =  jax.random.PRNGKey(0)

    #     def get_t_state(t_module):
    #         return {
    #             'params': {
    #                 'weight': t_to_np(t_module.weight),
    #                 'bias': t_to_np(t_module.bias)
    #             },
    #             'constants': {}
    #         }

    #     module_gen = lambda r: nn.LayerNorm(300, r)
    #     t_module_gen = lambda : torch.nn.LayerNorm(300)

    #     test_initialization(rng, module_gen, t_module_gen, get_t_state, 100)

    #     _, apply = nn.LayerNorm(3, rng)
    #     t_module = torch.nn.LayerNorm(3)
    #     state = get_t_state(t_module)
    #     x = jnp.array([[1,2,3],[5,6,6],[7,8,9]], dtype=float)
    #     x_t = torch.tensor(np.array(x))

    #     y, state = apply(state, x)
    #     y2, _ = apply(state, x)
    #     y_t = t_module(x_t).detach().numpy()

    #     self.assertTrue(allclose(y_t, y))
    #     self.assertTrue(allclose(y_t, y2))
 

    # def test_rngstate(self):
    #     rng = jax.random.PRNGKey(0)

    #     samples = []
    #     num_samples=10000

    #     with rng_util.RNGState(rng):
    #         for _ in range(num_samples):
    #             r = rng_util.split()
    #             samples.append(jax.random.normal(r))
    #     samples = jnp.array(samples)

    #     var = jnp.mean(samples**2 - jnp.mean(samples)**2)


    #     self.assertTrue(jnp.abs(var-1.0)<0.05)

    # def test_nested_modules(self):
    #     rng = jax.random.PRNGKey(0)
    #     # state, apply = NextModule(5, 10, 20, 2, 10, 20, rng=rng)#, return_torch=True)

    #     state, apply = NextModule(5, 10, 20, 2, 10, 20, rng=rng)
    #     # print(tree_map(lambda x: x, state))
    #     print(tree_map(lambda z: z.shape, state))

        # _, apply = NextModule(5, 10, 20, 2, return_torch=False)

        # x = jnp.ones(10, dtype=int)
        # x_t = torch.tensor(np.array(x))

        # y = apply(state, x)

        # y_t = t_module(x_t).detach().numpy()


        # self.assertTrue(jnp.allclose(y_t, y))  



if __name__ == 'main':
    unittest.main()

