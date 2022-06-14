"""MWE demonstrating GaussianHMM not being recognized as a pytree.

Error message:
  File ".../tests/issue_mwe_hmm_pytree.py", line 39, in <module>
    out = vmap(dummy_fn, (None, 0))(hmm, np.arange(5))
  File ".../ssm_jax/hmm/models.py", line 155, in tree_unflatten
    return cls.from_unconstrained_params(children, aux_data)
  File ".../ssm_jax/hmm/models.py", line 319, in from_unconstrained_params
    initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
  File ".../tensorflow_probability/substrates/jax/bijectors/bijector.py", line 1368, in forward
    return self._call_forward(x, name, **kwargs)
  File ".../tensorflow_probability/substrates/jax/bijectors/bijector.py", line 1344, in _call_forward
    x = nest_util.convert_to_nested_tensor(
  File ".../tensorflow_probability/substrates/jax/internal/nest_util.py", line 476, in convert_to_nested_tensor
    return convert_fn((), value, dtype, dtype_hint, name=name)
  File ".../tensorflow_probability/substrates/jax/internal/nest_util.py", line 471, in convert_fn
    return tf.convert_to_tensor(value, dtype, dtype_hint, name=name)
  File ".../tensorflow_probability/python/internal/backend/jax/ops.py", line 163, in _convert_to_tensor
    ret = conversion_func(value, dtype=dtype)
  File ".../tensorflow_probability/python/internal/backend/jax/ops.py", line 215, in _default_convert_to_tensor
    inferred_dtype = _infer_dtype(value, np.float32)
  File ".../tensorflow_probability/python/internal/backend/jax/ops.py", line 191, in _infer_dtype
    raise ValueError(('Attempt to convert a value ({})'
ValueError: Attempt to convert a value (<object object at 0x7f40108e0d70>) with an unsupported type (<class 'object'>) to a Tensor.

Viable workaround (just a bit clunky):
  from functools import partial
  weighted_suff_stats = vmap(partial(dummy_fn, hmm))(vec)
"""

from jax import vmap, jit
import jax.numpy as np
import jax.random as jr
from ssm_jax.hmm.models import GaussianHMM

def dummy_fn(hmm, val):
    return val

num_states = 10
emission_dim = 2
hmm = GaussianHMM.random_initialization(jr.PRNGKey(0), num_states, emission_dim)

out = vmap(dummy_fn, (None, 0))(hmm, np.arange(5))