# TODO: check scaling
# - check scale_jax_tensor and unscale_jax_tensor on few toy examples
# - check that scaling by 1 is equivalent to identity
# - check that scaling composed with unscaling is equivalent to identity
# - check tensor_types_to_scale and tensor_types_not_to_scale attribute in class JaxScaler,
#   example:
# @jdc.dataclass
# class Titi(JaxTensor):
#     array1: jax.ndarray

# @jdc.dataclass
# class Toto(JaxTensor):
#     array1: jax.ndarray
#     titi: titi

# JaxScaler(tensor_types_not_to_scale=(Titi, )).scale(Toto(array1, Titi(array1)))


def test_tensor():
    pass
