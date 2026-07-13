"""Vectorization helpers with chunked evaluation and Equinox support."""

import jax
import equinox as eqx

from jax import numpy as jnp
from jax import lax
from typing import Callable, Sequence, Any
from functools import partial
from jaxtyping import DTypeLike


Shape = Sequence[int]
DType = DTypeLike


def _argnums_partial(fun, args, dyn_argnums):

    sentinel = object()
    args_template = [sentinel] * len(args)
    dyn_args = []

    for i, arg in enumerate(args):
        if i in dyn_argnums:
            dyn_args.append(arg)
        else:
            args_template[i] = arg

    def fun_partial(*new_dyn_args):

        arg_iter = iter(new_dyn_args)

        interpolated_args = tuple(
            next(arg_iter) if arg == sentinel else arg for arg in args_template
        )

        return fun(*interpolated_args)

    return fun_partial, dyn_args


def _transpose_vmap_output(y, oax):
    if oax is None or oax == 0:
        return y
    else:
        return jnp.moveaxis(y, 0, oax)


def _transpose_vmap_outputs(outputs, axes):  # What a mess this is

    if len(axes) == 1:
        axes, outputs = (axes,), (outputs,)
        unpack = True
    else:
        unpack = False

    assert len(outputs) == len(axes)

    out = tuple(
        jax.tree.map(lambda l: _transpose_vmap_output(l, oax), leaf)
        for leaf, oax in zip(outputs, axes)
    )

    return out[0] if unpack else out


def _to_shape(x: int | tuple) -> tuple:
    return (x,) if isinstance(x, int) else x


def vmap(
    fun: Callable,
    in_axes: int | Shape = 0,
    out_axes: int | Shape = 0,
    chunk_size: int | None = None,
    checkpoint: bool = False,
    *args,
    **kwargs
) -> Callable:
    """Vectorize a function over a batch dimension with optional memory-efficient chunking.

    This function wraps `jax.vmap` to add support for processing large batches in chunks
    using `lax.map`, reducing memory usage while maintaining batched computation semantics.
    When `chunk_size` is None, it delegates directly to `jax.vmap`.

    Parameters
    ----------
    fun : Callable
        The function to vectorize. Will be called with batched versions of arguments
        specified by `in_axes`.
    in_axes : int | Shape, optional
        Specifies which axes of input arguments are batched. Can be:
        - A single int: same batch axis for all arguments (default 0)
        - A tuple of ints or None: per-argument batch axes, None means not batched
        By default 0.
    out_axes : int | Shape, optional
        Specifies where to place the batch axis in outputs. Same convention as `in_axes`.
        By default 0.
    chunk_size : int | None, optional
        If provided, process the batch in chunks of this size using `lax.map`.
        Reduces peak memory usage at the cost of sequential processing within chunks.
        If None, use standard `jax.vmap`. By default None.
    *args
        Additional positional arguments passed to `jax.vmap`.
    **kwargs
        Additional keyword arguments passed to `jax.vmap`.

    Returns
    -------
    Callable
        A vectorized function that takes the same inputs as `fun` but accepts
        batched arrays along the specified axes.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from dftax.vmap import vmap

    Basic vectorization over the first axis (default):

    >>> def square(x):
    ...     return x ** 2
    >>> vmap_square = vmap(square)
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> vmap_square(x)  # doctest: +SKIP
    Array([1., 4., 9.], dtype=float64)

    Vectorizing functions with multiple arguments and selective batch axes:

    >>> def add(x, y):
    ...     return x + y
    >>> vmap_add = vmap(add, in_axes=(0, None))  # x is batched, y is broadcast
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> y = jnp.array(10.0)
    >>> vmap_add(x, y)  # doctest: +SKIP
    Array([11., 12., 13.], dtype=float64)

    Memory-efficient processing of large batches with chunking:

    >>> vmap_chunked = vmap(square, chunk_size=32)
    >>> large_x = jnp.arange(1000.0)
    >>> result = vmap_chunked(large_x)  # Processes in chunks of 32 elements
    """

    if chunk_size is None:
        return jax.vmap(fun, in_axes, out_axes, *args, **kwargs)

    in_axes = _to_shape(in_axes)
    argnums = tuple(i for i, ix in enumerate(in_axes) if ix is not None)

    if not set(in_axes).issubset((0, None)):
        _in_axes = [ix % len(in_axes) for ix in in_axes if ix is not None]

        def preprocess_dyn_args(dyn_args):
            return jax.tree.map(jnp.moveaxis, dyn_args, _in_axes, [0] * len(_in_axes))

    else:
        preprocess_dyn_args = lambda x: x

    if not set(_to_shape(out_axes)).issubset((0, None)):
        postprocess_output = _transpose_vmap_outputs
    else:
        postprocess_output = lambda x, *_: x

    def f_chunked(*args, **kwargs):

        f_partial, dyn_args = _argnums_partial(partial(fun, **kwargs), args, argnums)
        dyn_args = preprocess_dyn_args(dyn_args)

        mapped_fn = lambda args: f_partial(*args)
        if checkpoint:
            mapped_fn = jax.checkpoint(mapped_fn)

        out = lax.map(mapped_fn, dyn_args, batch_size=chunk_size)

        return postprocess_output(out, _to_shape(out_axes))

    return f_chunked


class _VmapWrapper(eqx.Module):

    _fun: Callable
    _in_axes: int | Shape
    _out_axes: int | Shape
    _chunk_size: int | None
    _is_leaf: Callable[[Any], bool] = eqx.is_inexact_array

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, *args, **kwargs):

        dynamic_args, static_args = eqx.partition(args, self._is_leaf)

        @partial(vmap, in_axes=self._in_axes, out_axes=self._out_axes, chunk_size=self._chunk_size)
        def vmap_aux(*dyn_args):
            args = eqx.combine(dyn_args, static_args)
            return self._fun(*args, **kwargs)

        return vmap_aux(*dynamic_args)


def filter_vmap(
    fun: Callable,
    in_axes: int | Shape = 0,
    out_axes: int | Shape = 0,
    chunk_size: int | None = None,
    is_leaf: Callable[[Any], bool] = eqx.is_inexact_array,
) -> Callable:
    """Vectorize a function while preserving Equinox module structure via filtering.

    This function wraps `vmap` but handles Equinox pytree modules correctly by
    separating inexact arrays (which are vmapped) from other pytree nodes (which
    are not vmapped). This allows vmapping functions that take or return Equinox
    modules while only batching over array parameters, not module structure.

    Parameters
    ----------
    fun : Callable
        The function to vectorize. Typically a method or function that operates on
        or returns Equinox modules. Will be called with arrays vmapped and other
        pytree nodes preserved.
    in_axes : int | Shape, optional
        Specifies which axes of array inputs are batched. Can be:
        - A single int: same batch axis for all arguments (default 0)
        - A tuple of ints or None: per-argument batch axes, None means not batched
        Non-array arguments are never vmapped. By default 0.
    out_axes : int | Shape, optional
        Specifies where to place the batch axis in array outputs. Same convention
        as `in_axes`. By default 0.
    chunk_size : int | None, optional
        If provided, process the batch in chunks of this size. Reduces peak memory
        usage at the cost of sequential processing. If None, use standard `jax.vmap`.
        By default None.
    is_leaf : Callable[[Any], bool], optional
        A function that determines which pytree nodes are considered "leaves"
        (i.e., not vmapped). By default, `equinox.is_inexact_array`, which treats
        inexact arrays (floats, complex) as vmapped and everything else (ints,
        booleans, modules, containers) as static.

    Returns
    -------
    Callable
        A vectorized function wrapper that vectorizes over arrays and PyTrees.

    Examples
    --------
    Vectorizing a function over a batch of Equinox module parameters:

    >>> class SimpleModel(eqx.Module):
    ...     weight: jnp.ndarray
    ...     def __call__(self, x):
    ...         return self.weight * x

    >>> def forward_pass(model, x):
    ...     return model(x)

    >>> vmap_forward = filter_vmap(forward_pass, in_axes=(0, 0))
    >>> # Batch of models (weight dimension batched)
    >>> weights = jnp.array([[1.0], [2.0], [3.0]])
    >>> models = jax.tree.map(lambda w: SimpleModel(weight=w), weights)
    >>> x_batch = jnp.array([[1.0], [1.0], [1.0]])
    >>> # result will have shape (3, 1) with values [1., 2., 3.]
    >>> result = vmap_forward(models, x_batch)  # doctest: +SKIP

    Efficient batched inference with memory constraints:

    >>> vmap_forward_chunked = filter_vmap(forward_pass, in_axes=(0, 0), chunk_size=16)
    >>> # Process a large batch of models in chunks to save memory
    >>> large_result = vmap_forward_chunked(large_model_batch, large_x_batch)  # doctest: +SKIP
    """
    return _VmapWrapper(fun, in_axes, out_axes, chunk_size, is_leaf)
