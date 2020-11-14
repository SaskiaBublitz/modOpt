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

__all__ = ['store_newBoxes', 'get_entry_from_npz_dict', 'store_time']

def store_time(dict_name, time, position):
    """ stores time in npz directory with name dict_name
    
    Args:
        :dict_name:         string with npz directory name
        :time:              float value of measured time
        :position:          integer with position number of time array in 
                            npz directory

    Returns                 None

    """
    store_list_in_npz_dict(dict_name, time, position)


def store_newBoxes(dict_name, model, iterNo):
    """ stores new boxes in npz directory with name dict_name
    
    Args:
        :dict_name:         string with npz directory name
        :model:             instance of class model
        :iterNo:            integer with iteration step for array position in 
                            npz directory

    Returns                 None

    """
    newBoxes = model.xBounds
    listNewBoxes = []
    for box in newBoxes:
        listNewBoxes.append(numpy.array(parallelization.convertMpiToList(box)))
    store_list_in_npz_dict(dict_name, listNewBoxes, iterNo)

        
def append_npz_dict(dict_name, array, allow_pickle=False):
    """ appends npz directory by array
    
    Args:
        :dict_name:         string with npz directory name
        :array:             numpy array to append in npz directory

    Returns                 None

    """
    arrays = load_npz_dict(dict_name, allow_pickle)
    arrays.append(array)
    numpy.savez_compressed(dict_name, *arrays)

    
def load_npz_dict(dict_name, allow_pickle=False):
    """ loads npz directory with name dict_name
    
    Args:
        :dict_name:         string with npz directory name

    Returns                 None

    """
    arrays = []
    dict_data = numpy.load(dict_name, allow_pickle=allow_pickle)
    for key in dict_data.keys():
       arrays.append(dict_data[key])
    return arrays


def get_entry_from_npz_dict(dict_name, position):
    """ gets entry by position from npz directory with name dict_name
    
    Args:
        :dict_name:         string with npz directory name
        :position:          integer with array's position in npz directory

    Returns                 numpy array 

    """
    return load_npz_dict(dict_name)[position]


def store_list_in_npz_dict(dict_name, list_obj, array_id, allow_pickle=False):
    """ stores a list list_obj in a npz directory with name dict_name
    
    Args:
        :dict_name:         string with npz directory name
        :list_obj:          list that shall be stored
        :array_id:          if the list is the first object to store (id=0) the
                            npz directory needs to created beforehand
                            
    Returns                 None
    
    """
    if array_id == 0: numpy.savez_compressed(dict_name, numpy.array(list_obj))
    else: append_npz_dict(dict_name, numpy.array(list_obj), allow_pickle)