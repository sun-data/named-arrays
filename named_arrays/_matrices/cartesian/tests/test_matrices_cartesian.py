import named_arrays._vectors.cartesian.tests.test_vectors_cartesian
import named_arrays._matrices.tests.test_matrices

__all__ = [
    'AbstractTestAbstractCartesianMatrixArray',
    'AbstractTestAbstractExplicitCartesianMatrixArray',
    'AbstractTestAbstractExplicitCartesianMatrixArrayCreation',
    'AbstractTestAbstractImplicitCartesianMatrixArray',
    'AbstractTestAbstractCartesianMatrixRandomSample',
    'AbstractTestAbstractCartesianMatrixUniformRandomSample',
    'AbstractTestAbstractCartesianMatrixNormalRandomSample',
    'AbstractTestAbstractParameterizedCartesianMatrixArray',
]


class AbstractTestAbstractCartesianMatrixArray(
    named_arrays._matrices.tests.test_matrices.AbstractTestAbstractMatrixArray,
    named_arrays._vectors.cartesian.tests.test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
):
    pass


class AbstractTestAbstractExplicitCartesianMatrixArray(
    AbstractTestAbstractCartesianMatrixArray,
    named_arrays._matrices.tests.test_matrices.AbstractTestAbstractExplicitMatrixArray,
):
    pass


class AbstractTestAbstractExplicitCartesianMatrixArrayCreation(
    named_arrays._matrices.tests.test_matrices.AbstractTestAbstractExplicitMatrixArrayCreation,
):
    pass


class AbstractTestAbstractImplicitCartesianMatrixArray(
    named_arrays._matrices.tests.test_matrices.AbstractTestAbstractImplicitMatrixArray,
):
    pass


class AbstractTestAbstractCartesianMatrixRandomSample(
    AbstractTestAbstractImplicitCartesianMatrixArray,
    named_arrays._matrices.tests.test_matrices.AbstractTestAbstractMatrixRandomSample,
):
    pass


class AbstractTestAbstractCartesianMatrixUniformRandomSample(
    AbstractTestAbstractCartesianMatrixRandomSample,
    named_arrays._matrices.tests.test_matrices.AbstractTestAbstractMatrixUniformRandomSample,
):
    pass


class AbstractTestAbstractCartesianMatrixNormalRandomSample(
    AbstractTestAbstractCartesianMatrixRandomSample,
    named_arrays._matrices.tests.test_matrices.AbstractTestAbstractMatrixNormalRandomSample,
):
    pass


class AbstractTestAbstractParameterizedCartesianMatrixArray(
    AbstractTestAbstractImplicitCartesianMatrixArray,
    named_arrays._matrices.tests.test_matrices.AbstractTestAbstractParameterizedMatrixArray,
):
    pass
