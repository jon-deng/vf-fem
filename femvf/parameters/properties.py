"""
Classes for definining property values

A property should be away of defining the parameter vectors sizes/shapes for a model.

This should be described by a shape tuple, like
('field variable or constant variable?', shape of the data (constant or field valued))

shapes = OrderedDict()
        for key, property_type in model.PROPERTY_TYPES.items():
            field_or_const, data_shape = property_type

            shape = None
            if field_or_const == 'field':
                shape = (model.scalar_fspace.dim(), *data_shape)
            elif field_or_const == 'const':
                shape = (*data_shape, )
            else:
                raise ValueError("uh oh")

            shapes[key] = shape
"""

import numpy as np

from ..linalg import general_vec_set

def property_size(field_size, prop_type):
    """
    Return the size of a vector for a fluid model

    Parameters
    ----------
    field_size : int
    property_type : tuple
    """
    field_or_const, data_shape = prop_type

    shape = None
    if field_or_const == 'field':
        shape = (field_size*np.prod(data_shape, dtype=np.intp), )
    elif field_or_const == 'const':
        shape = (np.prod(data_shape, dtype=np.intp), )
    else:
        raise ValueError("uh oh")

    # Interpret properties with 1 element as '0-D' np.ndarray
    # TODO: I'm not sure if this is a good choice or not
    # if shape == (1,):
    #     shape = ()
    return shape

def property_vecs(field_size, prop_types, prop_defaults=None):
    labels = tuple(prop_types.keys())
    vecs = [np.zeros(property_size(field_size, prop_descr)) for _, prop_descr in prop_types.items()]

    if prop_defaults is not None:
        for label, vec in zip(labels, vecs):
            general_vec_set(vec, prop_defaults[label])

    return vecs, labels
