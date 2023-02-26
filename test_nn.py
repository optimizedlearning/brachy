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

import structure_utils as su


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

def tree_close(a, b, tol=1e-4):
    return tree_reduce_max(
        tree_map(
            lambda x, y: 2*jnp.linalg.norm(x-y)/(jnp.linalg.norm(x) + jnp.linalg.norm(y)+1e-8), 
            a, b)
    ) < tol

def zeros_like(a):
    return tree_map(lambda x: jnp.zeros_like(x), a)

def tree_size(a):
    return tree_reduce(lambda x, y: x+ y.size, a, 0)






def MyModule(vocab, embed, dim1, dim2, dim3=1, rng=None):
    if rng is None:
        rng = rng_util.split(1)

    organizer = su.StateOrganizer()

    with rng_util.RNGState(rng):

        organizer.embed = nn.Embedding(vocab, embed)
        organizer.seq = nn.Sequential(nn.Linear(embed, dim1), nn.Linear(dim1, dim2))

        r = rng_util.split(1)

        mul = 1 + jax.random.normal(r, (dim2,))
        organizer.register_constant('mul', mul)

        organizer.fc2 = nn.Linear(dim2, dim3)

    return organizer.create_module(MyModule_apply)



def MyModule_apply(tree, global_config, x):

    module = su.StateOrganizer(tree, global_config)

    x = module.embed(x)
    x = module.seq(x)
    x = module.mul * x
    x = module.fc2(x)

    return module.get_state_update(), x

class T_MyModule(torch.nn.Module):

    def __init__(self,vocab, embed, dim1, dim2, dim3=1):
        super().__init__()
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




def NextModule(vocab, embed, dim_next, dim_out, dim1, dim2, rng=None):
    if rng is None:
        rng = rng_util.split(1)

    organizer = su.StateOrganizer()


    with rng_util.RNGState(rng):
        organizer.trunk = MyModule(vocab, embed, dim1, dim2, dim_next)


        r = rng_util.split(1)

        bias = jax.random.normal(r, (dim_next,))

        
        organizer.register_parameter('next_bias', bias)

        organizer.head = nn.Linear(dim_next, dim_out)

    return organizer.create_module(NextModule_apply)



def NextModule_apply(tree, global_config, x):
    module = su.StateOrganizer(tree, global_config)

    x = module.trunk(x)
    x = jax.nn.relu(x)
    x = module.next_bias + x
    x = jax.nn.relu(x)
    x = module.head(x)

    return module.get_state_update(), x


class T_NextModule(torch.nn.Module):

    def __init__(self, vocab, embed, dim_next, dim_out, dim1, dim2):
        super().__init__()
        self.trunk = T_MyModule(vocab, embed, dim1, dim2, dim_next)
        bias = torch.nn.Parameter(torch.normal(torch.zeros(dim_next)))
        self.register_parameter('next_bias', bias)
        self.head = torch.nn.Linear(dim_next, dim_out)

    def forward(self, x):
        x = self.trunk(x)
        x = torch.nn.functional.relu(x)
        x = self.next_bias + x
        x = torch.nn.functional.relu(x)
        x = self.head(x)

        return x


def get_nested_state(t_module):

    state = {
        'params': {},
        'constants': {},
        'submodules': {},
    }

    params = state['params']
    constants = state['constants']
    submodules = state['submodules']

    params['next_bias'] = t_to_np(t_module.next_bias)

    submodules['head'] = {
        'params': {
            'weight': t_to_np(t_module.head.weight),
            'bias': t_to_np(t_module.head.bias)
        },
        'constants': {},
        'submodules': {}
    }

    trunk = t_module.trunk
    submodules['trunk'] = {
        'params': {},
        'constants': {},
        'submodules': {}
    }
    submodules['trunk']['submodules']['embed'] = {
        'params': {
            'weight': t_to_np(trunk.embed.weight)
        },
        'constants': {},
        'submodules': {}
    }

    submodules['trunk']['submodules']['fc2'] = {
        'params': {
            'weight': t_to_np(trunk.fc2.weight),
            'bias': t_to_np(trunk.fc2.bias)
        },
        'constants': {},
        'submodules': {}
    }

    submodules['trunk']['submodules']['seq'] = {
        'params': {},
        'constants': {},
        'submodules': {}
    }
    submodules['trunk']['submodules']['seq']['submodules'] = {
        i: {
            'params': {
                'weight': t_to_np(s.weight),
                'bias': t_to_np(s.bias)
            },
            'constants': {},
            'submodules': {}
        } for i, s in enumerate(trunk.seq)
    }

    submodules['trunk']['constants']['mul'] = t_to_np(trunk.mul)
    
    return state



def test_initialization(rng, module_gen, t_module_gen, get_t_state, sample_num=1000):
    mean = None
    var = None

    base = None
    base_t = None
    for _ in range(sample_num):
        rng, subkey = jax.random.split(rng)
        state, global_config = module_gen(subkey)
        t_module = t_module_gen()
        t_state  = get_t_state(t_module)

        if mean is None:
            mean = zeros_like(state['params'])
            var = zeros_like(state['params'])
            base = zeros_like(state['params'])
            base_t = zeros_like(state['params'])


        # print("state:   ",tree_map(lambda x: x.shape, state['params']))
        # print("t_state: ",tree_map(lambda x: x.shape, t_state['params']))
        mean = tree_sub_accum(mean, state['params'], t_state['params'])
        var = tree_square_sub_accum(var, state['params'], t_state['params'])
        base = tree_square_accum(base, state['params'])
        base_t = tree_square_accum(base, t_state['params'])


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


    def test_batch_norm(self):

        tree, global_config = nn.BatchNorm(2)

        state, apply = su.bind_module(tree, global_config)

        t_module = torch.nn.BatchNorm2d(2, dtype=torch.float32)

        B = 3
        C = 2
        H = 4
        W = 5

        x = jnp.array(range(B*C*H*W), dtype=jnp.float32).reshape((B, C, H, W))
        x2 = jnp.sqrt(jnp.array(range( B*C*H*W), dtype=jnp.float32).reshape((B, C, H, W)))

        x_t = torch.tensor(np.array(x))
        x2_t = torch.tensor(np.array(x2))

        state, y1 = apply(state, x)
        state, y2 = apply(state, x)
        state, y3 = apply(state, x2)

        apply = apply.bind_global_config({'train_mode': False})
        state, y4 = apply(state, x)

        y1_t = t_to_np(t_module(x_t))
        y2_t = t_to_np(t_module(x_t))
        y3_t = t_to_np(t_module(x2_t))

        t_module.eval()

        y4_t = t_to_np(t_module(x_t))

        assert jnp.allclose(y1, y1_t, atol=1e-6)

        assert jnp.allclose(y2, y2_t, atol=1e-6)

        assert jnp.allclose(y3, y3_t, atol=1e-6)


        assert jnp.allclose(y4, y4_t, atol=1e-6)

    def test_identity(self):
        state, global_config = nn.Identity()

        x = jnp.array([[1,2,3],[5,6,6],[7,8,9]], dtype=float)

        t_module = torch.nn.Identity()

        x_t = torch.tensor(np.array(x))

        _, y = state['apply'](state, global_config, x)

        y_t = t_module(x_t).numpy()

        self.assertTrue(allclose(y_t, y))



    def test_linear(self):

        rng = jax.random.PRNGKey(0)

        def get_t_state(t_module):
            return su.fill_tree({
                'params': {
                    'weight': t_to_np(t_module.weight),
                    'bias': t_to_np(t_module.bias)
                }
            })


        module_gen = lambda r: nn.Linear(300, 4000, bias=True, rng=r)
        t_module_gen = lambda : torch.nn.Linear(300, 4000, bias=True)
        test_initialization(rng, module_gen, t_module_gen, get_t_state, 100)



        tree, global_config = nn.Linear(3, 2, bias=True, rng=rng)
        t_module = torch.nn.Linear(3, 2, bias=True)
        state = get_t_state(t_module)

        tree = su.merge_trees(tree, state, keys_to_override=['params', 'constants'])
        apply = tree['apply']

        x = jnp.array([[1,2,3],[5,6,6],[7,8,9]], dtype=jnp.float32)
        x_t = torch.tensor(np.array(x))


        state, y = apply(state, global_config, x)
        _, y2 = apply(state, global_config, x)
        y_t = t_module(x_t).detach().numpy()

        self.assertTrue(allclose(y_t, y))
        self.assertTrue(allclose(y_t, y2))


    def test_embedding(self):
        rng =  jax.random.PRNGKey(0)

        def get_t_state(t_module):
            return su.fill_tree({
                'params': {
                    'weight': t_to_np(t_module.weight)
                },
                'constants': {}
            })

        module_gen = lambda r: nn.Embedding(500, 1000, rng=r)
        t_module_gen = lambda : torch.nn.Embedding(500, 1000)
        test_initialization(rng, module_gen, t_module_gen, get_t_state, 100)

        tree, global_config = nn.Embedding(30, 10, rng=rng)
        t_module = torch.nn.Embedding(30, 10)
        state = get_t_state(t_module)

        tree = su.merge_trees(tree, state, keys_to_override=['params', 'constants'])

        apply = tree['apply']

        x = jnp.array([0, 2, 29, 7, 4])
        x_t = torch.tensor(np.array(x))

        tree, y = apply(tree, global_config, x)
        _, y2 = apply(tree, global_config, x)
        y_t = t_module(x_t).detach().numpy()

        self.assertTrue(allclose(y_t, y))
        self.assertTrue(allclose(y_t, y2))

    def test_sequential(self):
        rng =  jax.random.PRNGKey(0)
        def get_t_state_for_init_test(t_module):
            params = []
            for l in t_module:
                params.append({
                    'weight': t_to_np(l.weight),
                    'bias': t_to_np(l.bias)
                })
            state = {
                'params': params,
                'constants': {}
            }
            return state

        def get_t_state_for_apply_test(t_module):
            state = su.empty_tree()
            submodules = state['submodules']
            for i, l in enumerate(t_module):
                submodules[i] = su.fill_tree({
                    'params': {
                        'weight': t_to_np(l.weight),
                        'bias': t_to_np(l.bias),
                    }
                })
            return state
            
        def module_gen(r):
            with rng_util.RNGState(r):
                chain = [
                    nn.Linear(3, 1000),
                    nn.Linear(1000, 500),
                    nn.Linear(500, 50)
                ]
                tree, global_config = nn.Sequential(*chain)
            # we do some hackery here to make the testing code work the same...
            tree['params'] = [tree['submodules'][i]['params'] for i in range(len(tree['submodules']))]
            return tree, global_config

        def t_module_gen():
            return torch.nn.Sequential(*[
                torch.nn.Linear(3, 1000),
                torch.nn.Linear(1000, 500),
                torch.nn.Linear(500, 50)
            ])

        test_initialization(rng, module_gen, t_module_gen, get_t_state_for_init_test, 500)

        with rng_util.RNGState(rng):
            chain = [
                nn.Linear(3, 10),
                nn.Linear(10, 20),
                nn.Linear(20, 3)
            ]
            tree, global_config = nn.Sequential(*chain)

        t_module = torch.nn.Sequential(*[
            torch.nn.Linear(3, 10),
            torch.nn.Linear(10, 20),
            torch.nn.Linear(20, 3)
        ])
        state = get_t_state_for_apply_test(t_module)

        tree = su.merge_trees(tree, state, keys_to_override=['params', 'constants'])

        apply = tree['apply']

        x = jnp.array([[1,2,3],[5,6,6],[7,8,9]], dtype=float)
        x_t = torch.tensor(np.array(x))

        tree, y = apply(tree, global_config, x)
        tree, y2 = apply(tree, global_config, x)

        y_t = t_module(x_t).detach().numpy()

        self.assertTrue(allclose(y_t, y))
        self.assertTrue(allclose(y_t, y2))


        
    def test_layer_norm(self):
        rng =  jax.random.PRNGKey(0)

        def get_t_state(t_module):
            return su.fill_tree({
                'params': {
                    'weight': t_to_np(t_module.weight),
                    'bias': t_to_np(t_module.bias)
                },
                'constants': {
                    'eps': 1e-5
                }
            })

        module_gen = lambda r: nn.LayerNorm(300, rng=r)
        t_module_gen = lambda : torch.nn.LayerNorm(300)

        test_initialization(rng, module_gen, t_module_gen, get_t_state, 100)

        tree, global_config = nn.LayerNorm(3, rng=rng)
        t_module = torch.nn.LayerNorm(3)
        state = get_t_state(t_module)

        tree = su.merge_trees(tree, state, keys_to_override=['params', 'constants'])

        apply = tree['apply']

        x = jnp.array([[1,2,3],[5,6,6],[7,8,9]], dtype=float)
        x_t = torch.tensor(np.array(x))

        tree, y = apply(tree, global_config, x)
        _, y2 = apply(tree, global_config, x)
        y_t = t_module(x_t).detach().numpy()

        self.assertTrue(allclose(y_t, y))
        self.assertTrue(allclose(y_t, y2))
 



    def test_conv2d(self):
        rng = jax.random.PRNGKey(0)

        def get_t_state(t_module):
            return su.fill_tree({
                'params': {
                    'weight': t_to_np(t_module.weight),
                    'bias': t_to_np(t_module.bias)
                },
                'constants': {
                    'padding': t_module.padding,
                    'stride': t_module.stride,
                    'dilation': t_module.dilation,
                    'feature_group_count': t_module.groups,                    
                }
            })

        module_gen = lambda r: nn.Conv2d(30, 40, 50, padding='same', bias=True, rng=r)
        t_module_gen = lambda: torch.nn.Conv2d(30, 40, 50, padding='same', bias=True)
        test_initialization(rng, module_gen, t_module_gen, get_t_state, 100)


        tree, global_config = nn.Conv2d(3, 4, 5, padding='same', bias=True, rng=rng)
        t_module = torch.nn.Conv2d(3, 4, 5, padding='same', bias=True)
        state = get_t_state(t_module)

        tree = su.merge_trees(tree, state, keys_to_override=['params', 'constants'])

        apply = tree['apply']

        x = jnp.array(np.random.normal(np.ones((2, 3, 6,7))), dtype=jnp.float32)
        x_t = torch.tensor(np.array(x))

        tree, y = apply(tree, global_config, x)
        _, y2 = apply(tree, global_config, x)
        y_t = t_module(x_t).detach().numpy()

        self.assertTrue(allclose(y_t, y))
        self.assertTrue(allclose(y_t, y2))



    def test_rngstate(self):
        rng = jax.random.PRNGKey(0)

        samples = []
        num_samples=10000

        with rng_util.RNGState(rng):
            for _ in range(num_samples):
                r = rng_util.split()
                samples.append(jax.random.normal(r))
        samples = jnp.array(samples)

        var = jnp.mean(samples**2 - jnp.mean(samples)**2)


        self.assertTrue(jnp.abs(var-1.0)<0.05)

    def test_nested_modules(self):
        rng = jax.random.PRNGKey(0)
    
        tree, global_config = NextModule(5, 10, 20, 2, 10, 20, rng=rng)
        init_state, apply = su.bind_module(tree, global_config)

        t_module = T_NextModule(5, 10, 20, 2, 10, 20)

        state = get_nested_state(t_module)


    
        x = jnp.ones(10, dtype=int)
        x_t = torch.tensor(np.array(x))

        state, y = apply(state, x)
        _, y2 = apply(state, x)
        y_t = t_module(x_t).detach().numpy()

        self.assertTrue(allclose(y_t, y))
        self.assertTrue(allclose(y_t, y2))


    def test_dropout(self):
        rng = jax.random.PRNGKey(0)

        tree, global_config = nn.Dropout(0.25, rng)
        apply = tree['apply']


        x = {
            'h': jnp.ones((10,2)),
            'k': {
                'l': jnp.ones(5),
                'p': jnp.ones(9)
            }
        }

        sum_dropout = zeros_like(x)

        num_trials = 2000

        for _ in range(num_trials):
            tree, drop = apply(tree, global_config, x)
            sum_dropout = tree_map(
                lambda a, b: a + b, sum_dropout, drop
            )

        mean_dropout = tree_map(
            lambda z: z/num_trials, sum_dropout
        )

        assert tree_close(mean_dropout, x, tol=0.05), f"not close: {mean_dropout}, {x}"

        
        tree, no_train_dropout = apply(tree, {'train_mode': False}, x)

        assert tree_close(no_train_dropout, x, tol=1e-5), f"dropout still applied when in eval mode"


        x = jnp.ones(10000)

        tree, drop = apply(tree, global_config, x)

        assert jnp.allclose(jnp.mean(drop), 1.0, atol=0.01), f"dropout on large array failed! mean {jnp.mean(drop)}, drop: {drop}"


    def test_train_mode(self):

        def simplemodule(rng):
            organizer = su.StateOrganizer()

            organizer.mul = 5.0
            organizer.dropout = nn.Dropout(0.4, rng=rng)

            return organizer.create_module(simpleapply)

        def simpleapply(tree, global_config, x):
            module = su.StateOrganizer(tree, global_config)

            m = x * module.mul

            d = module.dropout(m)

            return module.get_state_update(), d
        
    
        tree, global_config = simplemodule(jax.random.PRNGKey(0))
        state, apply = su.bind_module(tree, global_config)
        x = jnp.ones(5)

        new_state, y = apply(state, x)

        newer_state, y2 = apply(new_state, x)
        
        apply = apply.bind_global_config({'train_mode': False})

        _, y_eval = apply(new_state, x)

        assert jnp.linalg.norm(y-5*x) > 1.0
        assert jnp.linalg.norm(y-5*y2) > 1.0
        assert jnp.allclose(y_eval, 5*x)


    def test_multiheadattention(self):

        # this test is mostly just re-implementing it in a different way...
        # I'm too lazy to write the pytorch equivalency check right now...

        rng = jax.random.PRNGKey(0)

        embed_dim = 10
        num_heads = 2
        bias = True
        k_dim = 5
        v_dim = 4

        tree, global_config = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            k_dim=k_dim,
            v_dim=v_dim,
            rng=rng
        )
        state, apply = su.bind_module(tree, global_config)

        B = 3
        T = 6
        head_size = embed_dim // num_heads
        q = jnp.reshape(jnp.arange(B*T*embed_dim), (B, T, embed_dim))
        k = jnp.reshape(jnp.arange(B*T*k_dim), (B, T, k_dim))
        v = jnp.reshape(jnp.arange(B*T*v_dim), (B, T, v_dim))

        _, y = apply(state, q, k, v)

        
        # manual implementation:
        submodules = state['submodules']

        q_p = q @ submodules['q_proj']['params']['weight'].T + submodules['q_proj']['params']['bias']
        k_p = k @ submodules['k_proj']['params']['weight'].T + submodules['k_proj']['params']['bias']
        v_p = v @ submodules['v_proj']['params']['weight'].T + submodules['v_proj']['params']['bias']

        # extract heads manually
        q_h = [q_p[:, :, :head_size], q_p[:, :, -head_size:]]
        k_h = [k_p[:, :, :head_size], k_p[:, :, -head_size:]]
        v_h = [v_p[:, :, :head_size], v_p[:, :, -head_size:]]

        batch_transpose_mm = lambda a, b: einops.einsum(a, b, 'b i j, b k j -> b i k')
        batch_mm = lambda a, b: einops.einsum(a, b, 'b i j, b j k -> b i k')



        logits = [batch_transpose_mm(q_h[0], k_h[0])/jnp.sqrt(head_size), batch_transpose_mm(q_h[1], k_h[1])/jnp.sqrt(head_size)]

        softmax = [jax.nn.softmax(logits[0], axis=-1), jax.nn.softmax(logits[1], axis=-1)]

        values = [batch_mm(softmax[0], v_h[0]), batch_mm(softmax[1], v_h[1])]

        y_check = einops.rearrange(values, 'n b t h -> b t (n h)', n=2)



        assert jnp.allclose(y, y_check)


    def test_causal_attention(self):

        rng = jax.random.PRNGKey(0)

        embed_dim = 10
        num_heads = 2
        bias = True

        tree, global_config = nn.CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            rng=rng
        )
        state, apply = su.bind_module(tree, global_config)

        B = 3
        T = 6
        head_size = embed_dim // num_heads
        x_np = np.reshape(np.arange(B*T*embed_dim), (B, T, embed_dim))
        x = jnp.array(x_np)
        
        x_np[:,4:,:] = 0
        x_zeroed = jnp.array(x_np)


        state, y = apply(state, x)
        state, y_zeroed = apply(state, x_zeroed)    

        assert jnp.allclose(y[:,:3,:], y_zeroed[:, :3,:])   

        assert not jnp.allclose(y[1,4,2], y_zeroed[1, 4,2])    



if __name__ == 'main':
    unittest.main()

