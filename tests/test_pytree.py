# TODO:
# - test pytree_to_axes on custom instance of ConvertibleToAxes
#   example:
# class Titi(ConvertibleToAxes):
#     array1: jax.ndarray

#     def convert_to_axes(self, axis: Optional[AxisType_contra]) -> Self:
#         self.array1 = "titi"

# class Toto(ConvertibleToAxes):
#     array1: jax.ndarray
#     titi: titi

#     def convert_to_axes(self, axis: Optional[AxisType_contra]) -> Self:
#         self.array1 = "toto"

# pytree_to_axes(Toto(array1, Titi(array1)))



def test_pytree():
    pass
