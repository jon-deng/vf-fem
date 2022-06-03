"""
These are some functions used to interface jax dict outputs and BlockArray

"""

from blockarray.labelledarray import flatten_array

def blockvec_to_dict(blockvec):
    return {key: subvec for key, subvec in blockvec.items()}

def flatten_nested_dict_values(dict_array, labels):
    return tuple([
        flatten_nested_dict_values(dict_array[axis_label], labels[1:]) if len(labels) > 1 else dict_array[axis_label]
        for axis_label in labels[0]
    ])

def flatten_nested_dict(dict_array, labels):
    nested_array = flatten_nested_dict_values(dict_array, labels)
    return flatten_array(nested_array)

