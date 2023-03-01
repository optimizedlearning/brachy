
import jax
from jax import numpy as jnp
from jax import lax

import numpy as np

from jax._src.typing import Array, ArrayLike, DType, DTypeLike
from typing import overload, Any, Callable, Literal, Optional, Sequence, Tuple, Union


def relu(x: Array) -> Array:
    return jnp.maximum(x, 0)




# the jax.nn.softmax function has some instabilities related to the where argument.
# This one doesn't
def softmax(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None) -> Array:
  r"""Softmax function.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Args:
    x : input array
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.
      Either an integer or a tuple of integers.
    where: Elements to include in the :code:`softmax`.
  """
  x_max = jnp.max(x, axis, where=where, initial=-jnp.inf, keepdims=True)
  unnormalized = jnp.exp(x - lax.stop_gradient(x_max))
  if where is not None:
    unnormalized = jnp.where(where, unnormalized, 0.0)
  return unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)


def softmax_cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, axis=None):
    """Computes softmax cross entropy between sets of logits and integer labels.
    Measures the probability error in discrete classification tasks in which
    the classes are mutually exclusive (each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.
    References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)
    Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Integers specifying the correct class for each input, with shape
        `[...]`.
    Returns:
    Cross entropy between each prediction and the corresponding target
    distributions, with shape `[...]`.
    """
    # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
    # we avoid subtracting the normalizer from all values, just from the values
    # for the correct labels.

    if axis is None:
        if input.ndim == 1:
            axis = 0
        else:
            axis = 1

    # # let's transpose to make the logits in the last axis:
    # input = input.transpose(axis, -1)

    C = input.shape[axis]
    
    if weight is not None:
        weight_shape = (1,) * axis + (input.shape[axis],) + (1,) * (input.ndim - axis-1)
        weight = weight.reshape(weight_shape)

    # def entropy(probs):
    #     return -jnp.sum(probs * jnp.log(probs), axis=axis)

    if isinstance(target, int) or target.ndim != input.ndim:

        no_ignore = jax.lax.stop_gradient(target!=ignore_index)
        logits_max = jnp.max(input, axis=axis, keepdims=True)#, where=no_ignore, initial=0.0)
        logits = input - jax.lax.stop_gradient(logits_max)

        broadcast_shape = logits.shape[:axis] + (1,) + logits.shape[axis+1:]

        log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=axis, where=no_ignore.reshape(broadcast_shape)))


        labels_no_ignore = jnp.where(no_ignore, target, 0)

        label_logits = jnp.take_along_axis(logits, labels_no_ignore[..., None], axis=axis)[..., 0]
        


        if label_smoothing !=0 or weight is not None:
            one_hot_labels = jax.nn.one_hot(labels_no_ignore, num_classes=C, axis=axis)
            target_probs = one_hot_labels * (1.0 - label_smoothing) + jnp.ones_like(one_hot_labels)/C * label_smoothing

            if weight is not None:
                target_probs = target_probs * weight
                log_normalizers = log_normalizers * jnp.sum(target_probs, axis=axis)

            losses = -(jnp.sum(target_probs * logits, where=no_ignore.reshape(broadcast_shape), axis=axis) - log_normalizers)# - entropy(target_probs))
        else:

            label_logits = jnp.take_along_axis(logits, labels_no_ignore[..., None], axis=axis)[..., 0]
            losses = log_normalizers - label_logits
        

        losses = jnp.where(no_ignore, losses, 0.0)
    else:
        target_probs = target * (1.0 - label_smoothing) + jnp.ones_like(target)/C * label_smoothing

        logits_max = jnp.max(input, axis=axis, keepdims=True)#, initial=0.0)
        logits = input - jax.lax.stop_gradient(logits_max)

        log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=axis))


        if weight is not None:
            target_probs = target_probs * weight
            log_normalizers = log_normalizers * jnp.sum(target_probs * weight, axis=axis)

        losses = -(jnp.sum(target_probs * logits, axis=axis) - log_normalizers)# + entropy(target_probs))

        
        no_ignore = None
    


    if reduction == 'none':
        return losses
    if reduction == 'mean':
        return jnp.mean(losses, where=no_ignore)
    if reduction == 'sum':
        return jnp.sum(losses, where=no_ignore)
        


    # # ignore_labels = jnp.where(no_ignore, labels, jnp.zeros_like(labels))

    # # total = jax.lax.stop_gradient(jnp.sum(no_ignore))

    # label_logits = jnp.take_along_axis(logits, ignore_labels[..., None], axis=-1)[..., 0]

    # # log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    # # return jnp.sum(jnp.where(no_ignore, log_normalizers - label_logits, jnp.zeros_like(labels)))/total






def gen_pool2d(reducer, input, init_value, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, count_include_pad=True):
    # pytorch requires divisor_override to be an int, I have no idea why...
    # anyway, we will not require that.

    assert count_include_pad, "we don't support count_include_pad=False yet"

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    if isinstance(stride, int):
        stride = (stride, stride)
    
    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if stride is None:
        stride = (kernel_size[0], kernel_size[1])
    

    *N, C, Hi, Wi = input.shape

    if ceil_mode:
        rounder = np.ceil
    else:
        rounder = np.floor


    # in some testing, it seems like pytorch actually only properly applies the rounder
    # to Ho rather than Wo.
    # Unlike the inconsistent variance estimations in the batch norm, I'm going to NOT
    # propogate this weird behavior...
    # I dunno what this ceil option is every useful for anyway...
    Ho = int(rounder((Hi + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0] + 1 ))
    Wo = int(rounder((Wi + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1] + 1 ))


    H_padding = (padding[0], Ho * stride[0] + dilation[0] * (kernel_size[0] - 1) - Hi-padding[0])
    W_padding = (padding[1], Wo * stride[1] + dilation[1] * (kernel_size[1] - 1) - Wi-padding[1])

    pad_config = [(0,0) for _ in range(input.ndim)]
    pad_config[-2] = H_padding
    pad_config[-1] = W_padding

    window_shape = [1 for _ in range(input.ndim)]
    window_shape[-2] = kernel_size[0]
    window_shape[-1] = kernel_size[1]


    strides_shape = [1 for _ in range(input.ndim)]
    strides_shape[-2] = stride[0]
    strides_shape[-1] = stride[1]

    dilation_shape = [1 for _ in range(input.ndim)]
    dilation_shape[-2] = dilation[0]
    dilation_shape[-1] = dilation[1]

    x = jax.lax.reduce_window(input, init_value, reducer, window_shape, strides_shape, pad_config, window_dilation=dilation_shape)


    return x



def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1):
    reducer = jax.lax.max

    x = gen_pool2d(reducer, input, -jnp.inf, kernel_size, stride, padding, dilation)
    return x


def avg_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, count_include_pad=True, divisor_override=None):
    # pytorch requires divisor_override to be an int, I have no idea why...
    # anyway, we will not require that.

    reducer = jax.lax.add
    x = gen_pool2d(reducer, input, 0.0, kernel_size, stride, padding, dilation, ceil_mode, count_include_pad)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if divisor_override is None:
        divisor = kernel_size[0] * kernel_size[1]
    else:
        divisor = divisor_override
    x = x/divisor
    return x




def __getattr__(name, value):
    return getattr(jax.nn, name)
