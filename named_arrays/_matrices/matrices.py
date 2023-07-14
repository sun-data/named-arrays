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
        rows = self.cartesian_nd.rows
        rows_iter = iter(rows)
        try:
            row_prototype = rows[next(rows_iter)]
            return all(set(rows[r].components) == set(row_prototype.components) for r in rows_iter)
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
    def prototype_row(self):
        """
        Return a vector of the same type as each row of the matrix with each component zeroed.
        """
        rows = self.rows
        if not self.is_consistent:
            raise ValueError(
                f"matrix rows must all have the same abstract type, got "
                f"{tuple(rows[c].type_abstract for c in rows)}"
            )
        prototype_row = rows[next(iter(rows))]

        if isinstance(prototype_row, AbstractMatrixArray):
            prototype_row = prototype_row.prototype_row

        row_dict = {}
        for c in prototype_row.components:
            component = prototype_row.components[c]
            if isinstance(component, na.AbstractVectorArray):
                row_dict[c] = component.prototype_vector
            else:
                row_dict[c] = 0

        return prototype_row.type_explicit.from_components(row_dict)

    @property
    def prototype_column(self):
        """
        Return a vector of the same type of each column of the matrix with each component zeroed.
        """
        column_dict = {}
        for c in self.components:
            component = self.components[c]
            if isinstance(component, AbstractMatrixArray):
                column_dict[c] = component.type_vector.from_components(dict.fromkeys(component.components, 0))
            else:
                column_dict[c] = 0

        return self.type_vector.from_components(column_dict)

    @property
    def matrix_transpose(self):
        rows = self.rows

        if not self.is_consistent:
            raise ValueError(
                f"matrix rows must all have the same abstract type, got "
                f"{tuple(rows[c].type_abstract for c in rows)}"
            )

        row_prototype = self.prototype_row
        column_prototype = self.prototype_column

        new_matrix_dict = {}
        for c in row_prototype.components:
            component = row_prototype.components[c]
            if isinstance(component, na.AbstractVectorArray):
                new_matrix_dict[c] = component.type_matrix.from_components(dict.fromkeys(component.components,
                                                                                         column_prototype))
            else:
                new_matrix_dict[c] = column_prototype

        prototype_matrix_transpose = row_prototype.type_matrix.from_components(new_matrix_dict)

        result = prototype_matrix_transpose.cartesian_nd
        nd_rows = self.cartesian_nd.rows
        for r in nd_rows:
            for c in nd_rows[r].components:
                value = nd_rows[r].components[c]
                result.components[c].components[r] = value

        return prototype_matrix_transpose.from_cartesian_nd(result, like=prototype_matrix_transpose)

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
        x1 = x1.explicit
        x2 = x2.explicit

        if isinstance(x1, na.AbstractMatrixArray):
            components_x1 = x1.explicit.cartesian_nd.components

            if isinstance(x2, na.AbstractMatrixArray):

                x2 = x2.matrix_transpose
                type_row = x2.type_vector
                components_x2 = x2.cartesian_nd.components

                result = dict()
                for r in components_x1:
                    result[r] = na.CartesianNdVectorArray(
                        {c: components_x1[r] @ components_x2[c] for c in components_x2}
                    )
                print(result)
                result = x1.from_cartesian_nd(na.CartesianNdMatrixArray(result), like=x1)

            else:
                if x1.type_vector().type_abstract == x2.type_abstract:
                    result_components = {r: components_x1[r] @ x2.cartesian_nd for r in components_x1}
                    result = x2.type_explicit.from_cartesian_nd(na.CartesianNdVectorArray(result_components), like=x2)

                else:
                    result = NotImplemented

        else:
            if isinstance(x2, na.AbstractMatrixArray):

                x2 = x2.matrix_transpose
                components_x2 = x2.cartesian_nd.components

                component_dict = {c: x1.cartesian_nd @ components_x2[c] for c in components_x2}
                result = x1.type_explicit.from_cartesian_nd(
                    na.CartesianNdVectorArray(component_dict), like=x1,
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

    @classmethod
    def from_cartesian_nd(
            cls: AbstractExplicitMatrixArray,
            cartesian_nd: na.CartesianNdMatrixArray,
            like: None | AbstractExplicitMatrixArray = None,
    ) -> AbstractExplicitMatrixArray:

        if like is None:
            components_new = cartesian_nd.components

        else:
            nd_components = cartesian_nd.components
            components_new = {}
            components = like.components
            for c in components:

                component = components[c]

                if isinstance(component, na.AbstractMatrixArray):
                    secondary_components = {}
                    for c2 in component.components:
                        component2 = component.components[c2]

                        secondary_components[c2] = component2.type_explicit.from_cartesian_nd(
                            nd_components[f"{c}_{c2}"],
                            like=component2)

                    components_new[c] = component.type_explicit.from_components(secondary_components)
                else:
                    components_new[c] = component.type_explicit.from_cartesian_nd(
                        nd_components[c],
                        like=component)

        return cls.from_components(components_new)


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
