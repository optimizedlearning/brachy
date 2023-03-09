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

import functional as F


def t_to_np(tensor):
    return tensor.detach().numpy()


class TestFunctional(unittest.TestCase):

    def test_chain(self):
        rng = jax.random.PRNGKey(0)
        lin = nn.Linear(5,5, bias=False, rng=rng)
        lin[0]['params']['weight'] = 2*jnp.eye(5)

        tree, g_config = F.chain(lin, lambda x: x**2, lambda x: jnp.sum(x))

        x = jnp.ones(5)

        _, y = tree['apply'](tree, g_config, x)

        assert jnp.allclose(y, 20)



    def test_accuracy(self):
        x = jnp.array([[1.0, 4.2, 3.4], [0.2,12,1.0], [-2,100,-2]], dtype=float)
        labels = jnp.array([0,1, -100], dtype=int)


        weights = jnp.array([1,2,3])

        y = F.accuracy(x, labels)

        assert(jnp.allclose(0.5, y))

        y = F.accuracy(x, labels, weight=weights)

        assert(jnp.allclose(2.0/3.0, y))

    def test_softmax_cross_entropy(self):
        x = jnp.array([[1.0, 4.2, 3.4], [0.2,-12,1.0], [-2,100,-2]], dtype=float)
        labels = jnp.array([0,1, -100], dtype=int)

        x_t = torch.tensor(np.array(x))
        labels_t = torch.tensor(np.array(labels), dtype=torch.long)

        y = F.softmax_cross_entropy(x, labels, reduction='none')
        y_t = t_to_np(torch.nn.functional.cross_entropy(x_t, labels_t , reduction='none'))

        assert jnp.allclose(y, y_t), f"failed no reduction test:\n{y}\n{y_t}"


        y = F.softmax_cross_entropy(x, labels)
        y_t = t_to_np(torch.nn.functional.cross_entropy(x_t, labels_t))

        assert jnp.allclose(y, y_t), f"failed mean/default test:\n{y}\n{y_t}"

        y = F.softmax_cross_entropy(x, labels, reduction='sum')
        y_t = t_to_np(torch.nn.functional.cross_entropy(x_t, labels_t , reduction='sum'))

        assert jnp.allclose(y, y_t), f"failed sum test:\n{y}\n{y_t}"

        y = F.softmax_cross_entropy(x, labels, label_smoothing=0.1, reduction='none')
        y_t = t_to_np(torch.nn.functional.cross_entropy(x_t, labels_t, label_smoothing=0.1, reduction='none'))

        assert jnp.allclose(y, y_t), f"failed smoothed test:\n{y}\n{y_t}"





    def test_softmax_cross_entropy_probs(self):
        x = jnp.array([[1.0, 4.2, 3.4], [0.2,-12,1.0], [-2,100,-2]], dtype=float)
        labels = jnp.array([[1.0, 0.0, 0.0], [0.0,1.0,0.0], [0.0, 0.0, 1.0]], dtype=float)

        x_t = torch.tensor(np.array(x))
        labels_t = torch.tensor(np.array(labels))


        y = F.softmax_cross_entropy(x, labels, label_smoothing=0.01)
        y_t = t_to_np(torch.nn.functional.cross_entropy(x_t, labels_t, label_smoothing=0.01))

        assert jnp.allclose(y, y_t), f"failed targets as prob test:\n{y}\n{y_t}"


    def test_softmax_cross_entropy_other_axis(self):
        x = jnp.array(range(2*3*4*5), dtype=float).reshape((4,5,3,2))
        labels = jnp.array([0,1,-23,2] * 2*5, dtype=int).reshape(4,5,2)

        x_t = torch.tensor(np.array(x)).transpose(1,2)
        labels_t = torch.tensor(np.array(labels), dtype=torch.long)

        y = F.softmax_cross_entropy(x, labels, reduction='mean', axis=2, ignore_index=-23)
        y_t = t_to_np(torch.nn.functional.cross_entropy(x_t, labels_t , reduction='mean', ignore_index=-23))

        assert jnp.allclose(y, y_t), f"failed no reduction test:\n{y}\n{y_t}"


    def test_softmax_cross_entropy_weight(self):
        x = jnp.array([[1.0, 4.2, 3.4], [0.2,-12,1.0], [-2,100,-2]], dtype=float)
        labels = jnp.array([0,1, -100], dtype=int)
        weight = jnp.array([1.0,2.0,3.0], dtype=float)

        x_t = torch.tensor(np.array(x))
        labels_t = torch.tensor(np.array(labels), dtype=torch.long)
        weight_t = torch.tensor(np.array(weight))

        y = F.softmax_cross_entropy(x, labels, reduction='none', weight=weight)
        y_t = t_to_np(torch.nn.functional.cross_entropy(x_t, labels_t , reduction='none', weight=weight_t))

        assert jnp.allclose(y, y_t), f"failed no reduction test:\n{y}\n{y_t}"


    def test_avgpool2d(self):

        x = jnp.array(range(4*5*8*10), dtype=float).reshape((4,5,8,10))

        x_t = torch.tensor(np.array(x))

        y = F.avg_pool2d(x, 3)
        y_t = t_to_np(torch.nn.functional.avg_pool2d(x_t, 3))

        assert jnp.allclose(y, y_t)


        y = F.avg_pool2d(x, (4,2), (2,3), padding=(2,1))
        y_t = t_to_np(torch.nn.functional.avg_pool2d(x_t, (4,2), (2,3), padding=(2,1)))

        assert jnp.allclose(y, y_t)


        y = F.avg_pool2d(x, (4,2), (2,3), padding=(2,1), divisor_override=4.0)
        y_t = t_to_np(torch.nn.functional.avg_pool2d(x_t, (4,2), (2,3), padding=(2,1), divisor_override=4))

        assert jnp.allclose(y, y_t)



    def test_maxpool2d(self):

        x = jnp.array(range(4*5*8*10), dtype=float).reshape((4,5,8,10))

        x_t = torch.tensor(np.array(x))

        y = F.max_pool2d(x, 3)
        y_t = t_to_np(torch.nn.functional.max_pool2d(x_t, 3))

        assert jnp.allclose(y, y_t)


        y = F.max_pool2d(x, (4,2), (2,3), padding=(2,1))
        y_t = t_to_np(torch.nn.functional.max_pool2d(x_t, (4,2), (2,3), padding=(2,1)))

        assert jnp.allclose(y, y_t)


        y = F.max_pool2d(x, (4,2), (2,3), padding=(2,1), dilation=(1,2))
        y_t = t_to_np(torch.nn.functional.max_pool2d(x_t, (4,2), (2,3), padding=(2,1), dilation=(1,2)))

        assert jnp.allclose(y, y_t)

