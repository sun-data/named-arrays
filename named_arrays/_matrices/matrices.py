from __future__ import annotations
from typing import Type
import abc
import dataclasses
import named_arrays as na

__all__ = [
    'AbstractMatrixArray',
    'AbstractExplicitMatrixArray',
    'AbstractImplicitMatrixArray',
    'AbstractMatrixRandomSample',
    'AbstractMatrixUniformRandomSample',
    'AbstractMatrixNormalRandomSample',
    'AbstractParameterizedMatrixArray',
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMatrixArray(
    na.AbstractVectorArray,
):

    @property
    @abc.abstractmethod
    def type_vector(self) -> Type[na.AbstractExplicitVectorArray]:
        """
        The corresponding :class:`named_arrays.AbstractVectorArray` class
        """

    @property
    def components(self) -> dict[str, na.AbstractVectorArray]:
        return super().components

    @property
    def entries(self) -> dict[tuple[str, str], na.ScalarLike]:
        rows = self.rows
        return {(r, c): rows[r].components[c] for r in rows for c in rows[r].components}

    @property
    def rows(self) -> dict[str, na.AbstractVectorArray]:
        """
        Rows of the matrix, same as :attr:`components`
        """
        return self.components

    @property
    def is_consistent(self) -> bool:
        """
        Check if all the rows of the matrix have the same abstract type.
        """
        rows = self.rows
        rows_iter = iter(rows)
        try:
            row_prototype = rows[next(rows_iter)].type_abstract
            return all(rows[r].type_abstract == row_prototype for r in rows_iter)
        except AttributeError:
            return False

    @property
    def is_square(self):
        """
        Check if this matrix has the same number of rows and columns.
        """
        if self.is_consistent:
            rows = self.rows
            num_rows = len(rows)
            return all(num_rows == len(rows[r].components) for r in rows)
        else:
            return False

    @property
    def matrix_transpose(self):
        rows = self.rows
        row_prototype = rows[next(iter(rows))]
        if not self.is_consistent:
            raise ValueError(
                f"matrix rows must all have the same abstract type, got "
                f"{tuple(rows[c].type_abstract for c in rows)}"
            )
        type_matrix = row_prototype.type_matrix
        type_row = self.type_vector
        # row_dict = {c: type_row() for c in row_prototype.components}
        row_dict = {c: type_row.from_components(dict.fromkeys(row_prototype.components, 0)) for c in
                    row_prototype.components}
        result = type_matrix.from_components(row_dict)

        for r in rows:
            for c in rows[r].components:
                result.components[c].components[r] = rows[r].components[c]

        return result

    @property
    @abc.abstractmethod
    def determinant(self) -> na.ScalarLike:
        """
        The determinant of this matrix
        """

    @property
    @abc.abstractmethod
    def inverse(self) -> AbstractMatrixArray:
        """
        The inverse of this matrix
        """

    @property
    def cartesian_nd(self):
        """
        Convert all cartesian vectors making up the matrix to instances of :class:`AbstractCartesianNdVectorArray`
        """
        components_new = dict()
        components = self.components
        for c in components:
            component = components[c]

            if isinstance(component, na.AbstractMatrixArray):
                for c2 in component.components:
                    components_new[f"{c}_{c2}"] = component.components[c2].cartesian_nd
            else:
                components_new[c] = component.cartesian_nd

        return na.CartesianNdMatrixArray(components_new)

    def __array_matmul__(
            self,
            x1: na.ArrayLike,
            x2: na.ArrayLike,
            out: tuple[None | na.AbstractExplicitArray] = (None,),
            **kwargs,
    ) -> na.AbstractExplicitArray:

        if isinstance(x1, na.AbstractMatrixArray):

            components_x1 = x1.components

            if isinstance(x2, na.AbstractMatrixArray):

                x2 = x2.matrix_transpose
                type_row = x2.type_vector
                components_x2 = x2.components

                result = dict()
                for r in components_x1:
                    result[r] = type_row.from_components(
                        {c: components_x1[r] @ components_x2[c] for c in components_x2}
                    )
                result = x1.type_matrix.from_components(result)

            else:
                result = x1.type_vector.from_components(
                    {r: components_x1[r] @ x2 for r in components_x1}
                )

        else:
            if isinstance(x2, na.AbstractMatrixArray):

                x2 = x2.matrix_transpose
                components_x2 = x2.components

                result = x2.type_vector.from_components(
                    {c: x1 @ components_x2[c] for c in components_x2}
                )

            else:
                result = NotImplemented

        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractExplicitMatrixArray(
    AbstractMatrixArray,
    na.AbstractExplicitVectorArray,
):

    @property
    def components(self) -> dict[str, na.AbstractVectorArray]:
        return self.__dict__

    @components.setter
    def components(self, value: dict[str, na.AbstractVectorArray]):
        self.__dict__ = value


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitMatrixArray(
    AbstractMatrixArray,
    na.AbstractImplicitVectorArray
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMatrixRandomSample(
    AbstractImplicitMatrixArray,
    na.AbstractVectorRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMatrixUniformRandomSample(
    AbstractMatrixRandomSample,
    na.AbstractVectorUniformRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMatrixNormalRandomSample(
    AbstractMatrixRandomSample,
    na.AbstractVectorUniformRandomSample,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractParameterizedMatrixArray(
    AbstractImplicitMatrixArray,
    na.AbstractParameterizedVectorArray,
):
    pass
