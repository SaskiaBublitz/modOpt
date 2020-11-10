""" Methods to store data """

"""
***************************************************
Import packages
***************************************************
"""
from modOpt.constraints import parallelization
import numpy

"""
****************************************************
Methods to store boxes, sample points and other data
****************************************************
"""

__all__ = [ 'storeNewBoxesInNPZ', 'get_entry_from_npz_dict']

def storeNewBoxesInNPZ(dict_name, model, iterNo):
    newBoxes = model.xBounds
    listNewBoxes = []
    for box in newBoxes:
        listNewBoxes.append(numpy.array(parallelization.convertMpiToList(box)))
    store_list_in_npz_dict(dict_name, listNewBoxes, iterNo-1)

        
def append_npz_dict(dict_name, array, allow_pickle=False):
    arrays = load_npz_dict(dict_name, allow_pickle)
    arrays.append(array)
    numpy.savez_compressed(dict_name, *arrays)

    
def load_npz_dict(dict_name, allow_pickle=False):
    arrays = []
    dict_data = numpy.load(dict_name, allow_pickle=allow_pickle)
    for key in dict_data.keys():
       arrays.append(dict_data[key])
    return arrays


def get_entry_from_npz_dict(dict_name, position):
    return load_npz_dict(dict_name)[position]


def store_list_in_npz_dict(dict_name, list_obj, array_id, allow_pickle=False):
    if array_id == 0: numpy.savez_compressed(dict_name, numpy.array(list_obj))
    else: append_npz_dict(dict_name, numpy.array(list_obj), allow_pickle)