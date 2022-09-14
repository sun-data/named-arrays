import named_arrays as na


def test_broadcast_shapes():
    num_x = 5
    num_y = 7
    num_z = 11

    shape_1 = dict(x=num_x)
    shape_2 = dict(x=num_x, y=num_y)
    shape_3 = dict(y=num_y, z=num_z)

    shape_broadcasted = na.broadcast_shapes(shape_1, shape_2, shape_3)
    assert shape_broadcasted == dict(x=num_x, y=num_y, z=num_z)


def test_shape_broadcasted():

    num_x = 5
    num_y = 7
    num_z = 11

    shape_1 = dict(x=num_x)
    shape_2 = dict(x=num_x, y=num_y)
    shape_3 = dict(y=num_y, z=num_z)

    array_1 = na.ScalarArray.empty(shape_1)
    array_2 = na.ScalarArray.empty(shape_2)
    array_3 = na.ScalarArray.empty(shape_3)

    shape_broadcasted = na.shape_broadcasted(array_1, array_2, array_3)
    assert shape_broadcasted == dict(x=num_x, y=num_y, z=num_z)
