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
    def entries(self) -> dict[tuple[str, ...], na.ScalarLike]:
        rows = self.cartesian_nd.rows
        result = {}
        for r in rows:
            row = rows[r]
            if isinstance(row, na.AbstractArray):
                for c in row.components:
                    result[(r, c)] = row.components[c]
            else:
                result[(r,)] = row

        return result

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
            rows = self.cartesian_nd.rows
            num_rows = len(rows)
            return all(num_rows == len(rows[r].components) for r in rows)
        else:
            return False

    @property
    def matrix(self) -> AbstractMatrixArray:
        return self

    @property
    def row_prototype(self) -> na.AbstractVectorArray:
        """
        Return a vector of the same type as each row of the matrix.
        """
        rows = self.rows
        if not self.is_consistent:
            raise ValueError(
                f"matrix rows must all have the same abstract type, got "
                f"{tuple(rows[c].type_abstract for c in rows)}"
            )
        prototype_row = rows[next(iter(rows))]

        if isinstance(prototype_row, AbstractMatrixArray):
            prototype_row = prototype_row.row_prototype

        return prototype_row

    @property
    def column_prototype(self) -> na.AbstractMatrixArray:
        """
        Return a vector representing a column of the matrix with each component zeroed.
        """
        column_dict = {}
        for c in self.components:
            component = self.components[c]
            if isinstance(component, AbstractMatrixArray):
                column_dict[c] = component.column_prototype
            else:
                column_dict[c] = 0

        return self.type_vector.from_components(column_dict)

    def prototype_matrix(self, row: na.AbstractVectorArray = None):
        if row is None:
            row = self.row_prototype

        new_dict = {}
        components = self.column_prototype.matrix.components
        for c in components:
            component = components[c]
            if isinstance(component, AbstractMatrixArray):
                new_dict[c] = component.prototype_matrix(row)
            else:
                new_dict[c] = row

        return self.type_explicit.from_components(new_dict)

    @property
    def matrix_transpose(self):
        rows = self.rows

        if not self.is_consistent:
            raise ValueError(
                f"matrix rows must all have the same abstract type, got "
                f"{tuple(rows[c].type_abstract for c in rows)}"
            )

        new_row = self.column_prototype
        new_column = self.row_prototype.matrix
        # new_row = self.matrix_transpose.row_prototype

        prototype_matrix_transpose = new_column.prototype_matrix(new_row)

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
    def inverse(self) -> AbstractMatrixArray:
        """
        The inverse of this matrix
        """
        explicit = self.explicit

        a = explicit.cartesian_nd

        if not a.is_square:
            raise ValueError("can only invert square matrices")

        rows = a.rows

        if len(rows) == 1:
            r_1, = rows.keys()
            row_1, = [r.components for r in rows.values()]
            c_1, = row_1.keys()
            a, = row_1.values()
            result = na.CartesianNdMatrixArray({
                c_1: na.CartesianNdVectorArray({
                    r_1: 1 / a
                })
            })

        elif len(rows) == 2:
            r_1, r_2 = rows.keys()
            row_1, row_2 = [r.components for r in rows.values()]
            c_1, c_2 = row_1.keys()
            a, b = row_1.values()
            c, d = row_2.values()
            det = a * d - b * c
            result = na.CartesianNdMatrixArray({
                c_1: na.CartesianNdVectorArray({
                    r_1: d / det,
                    r_2: -b / det,
                }),
                c_2: na.CartesianNdVectorArray({
                    r_1: -c / det,
                    r_2: a / det,
                }),
            })

        elif len(rows) == 3:
            r_1, r_2, r_3 = rows.keys()
            row_1, row_2, row_3 = [r.components for r in rows.values()]
            c_1, c_2, c_3 = row_1.keys()
            a, b, c = row_1.values()
            d, e, f = row_2.values()
            g, h, i = row_3.values()
            det = (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h)
            result = na.CartesianNdMatrixArray({
                c_1: na.CartesianNdVectorArray({
                    r_1: (e * i - f * h) / det,
                    r_2: -(b * i - c * h) / det,
                    r_3: (b * f - c * e) / det,
                }),
                c_2: na.CartesianNdVectorArray({
                    r_1: -(d * i - f * g) / det,
                    r_2: (a * i - c * g) / det,
                    r_3: -(a * f - c * d) / det,
                }),
                c_3: na.CartesianNdVectorArray({
                    r_1: (d * h - e * g) / det,
                    r_2: -(a * h - b * g) / det,
                    r_3: (a * e - b * d) / det,
                }),
            })

        else:
            unit = na.unit(a, squeeze=False)
            value = na.value(a)
            value = na.stack([
                na.stack(list(row.components.values()), axis="_column")
                for row in value.rows.values()
            ], axis="_row")
            inverse = value.matrix_inverse(axis_rows="_row", axis_columns="_column")
            result = 1 / unit.matrix_transpose
            for i, r in enumerate(result.rows):
                row = result.rows[r].components
                for j, c in enumerate(row):
                    row[c] = inverse[dict(_row=i, _column=j)] * row[c]

        result = explicit.from_cartesian_nd(result, like=explicit.matrix_transpose)

        return result

    @property
    def cartesian_nd(self) -> na.AbstractCartesianNdMatrixArray:
        """
        Convert all cartesian vectors making up the matrix to instances of :class:`AbstractCartesianNdVectorArray`
        """
        components_new = dict()
        components = self.components
        for c in components:
            component = components[c]

            if isinstance(component, na.AbstractMatrixArray):
                component2 = component.cartesian_nd.components
                for c2 in component2:
                    components_new[f"{c}_{c2}"] = component2[c2]
            elif isinstance(component, na.AbstractVectorArray):
                components_new[c] = component.cartesian_nd
            else:
                components_new[c] = component

        return na.CartesianNdMatrixArray(components_new)

    def __array_matmul__(
            self,
            x1: na.ArrayLike,
            x2: na.ArrayLike,
            out: tuple[None | na.AbstractExplicitArray] = (None,),
            **kwargs,
    ) -> na.AbstractExplicitArray:

        if isinstance(x1, na.AbstractMatrixArray):
            components_x1 = x1.cartesian_nd.components

            if isinstance(x2, na.AbstractMatrixArray):
                if x1.row_prototype.cartesian_nd.components.keys() == x2.column_prototype.cartesian_nd.components.keys():

                    prototype_matrix = x1.prototype_matrix(x2.row_prototype)

                    x2 = x2.matrix_transpose
                    components_x2 = x2.cartesian_nd.components

                    result = dict()
                    for r in components_x1:
                        result[r] = na.CartesianNdVectorArray(
                            {c: components_x1[r] @ components_x2[c] for c in components_x2}
                        )
                    result = prototype_matrix.from_cartesian_nd(
                        na.CartesianNdMatrixArray(result),
                        like=prototype_matrix,
                    )
                else:
                    result = NotImplemented

            elif isinstance(x2, na.AbstractVectorArray):
                if x1.row_prototype.type_abstract == x2.type_abstract:
                    result_components = {r: components_x1[r] @ x2.cartesian_nd for r in components_x1}
                    result = x1.type_vector.from_cartesian_nd(
                        array=na.CartesianNdVectorArray(result_components),
                        like=x1.column_prototype,
                    )

                else:
                    result = NotImplemented
            else:
                result_components = {r: components_x1[r] @ x2 for r in components_x1}
                result = x1.type_explicit.from_cartesian_nd(na.CartesianNdMatrixArray(result_components), like=x1)

        else:
            if isinstance(x2, na.AbstractMatrixArray):

                x2 = x2.matrix_transpose
                components_x2 = x2.cartesian_nd.components

                if isinstance(x1, na.AbstractVectorArray):
                    component_dict = {c: x1.cartesian_nd @ components_x2[c] for c in components_x2}
                    result = x1.type_explicit.from_cartesian_nd(na.CartesianNdVectorArray(component_dict), like=x1)
                else:
                    component_dict = {c: x1 @ components_x2[c] for c in components_x2}
                    result = x2.type_explicit.from_cartesian_nd(na.CartesianNdMatrixArray(component_dict), like=x2)

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
        cls,
        array: na.CartesianNdMatrixArray,
        like: None | AbstractExplicitMatrixArray = None
    ) -> AbstractExplicitMatrixArray:

        if like is None:
            components_new = array.components

        else:
            nd_components = array.components
            components_new = {}
            components = like.components
            for c in components:

                component = components[c]
                if isinstance(component, na.AbstractVectorArray):
                    if isinstance(component, na.AbstractMatrixArray):
                        nd_key_mod = f"{c}_"
                        sub_dict = {k[len(nd_key_mod):]: v for k, v in nd_components.items() if k.startswith(nd_key_mod)}
                        components_new[c] = component.type_explicit.from_cartesian_nd(
                             na.CartesianNdMatrixArray(sub_dict),
                             like=component,
                         )
                    else:
                        components_new[c] = component.type_explicit.from_cartesian_nd(nd_components[c], like=component)
                else:
                    components_new[c] = nd_components[c]

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
