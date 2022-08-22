"""
***************************************************
Import packages
***************************************************
"""
import numpy
import mpmath
import modOpt.storage as mostg
from modOpt.constraints import iNes_procedure
"""
***************************************************
sets initials to arithemtic mean value of an interval
***************************************************
"""

__all__ = ['setStateVarValuesToMidPointOfIntervals', 'set_close_to_bound']

def set_close_to_bound(res_solver, bxrd_options):
    if "Model" in res_solver.keys():
        model = res_solver["Model"]
        
        if model.xBounds !=[]:    
            model.stateVarValues = numpy.empty((len(model.xBounds), 
                                                len(model.xBounds[0])))
            for k, box in enumerate(model.xBounds):
                r = model.complete_parent_boxes[k][0]
                box_ID = model.complete_parent_boxes[k][1]
                box_new_float=[[iNes_procedure.convert_mpi_float(iv.a), 
                                iNes_procedure.convert_mpi_float(iv.b)] 
                               for iv in box]
                box_old = mostg.get_entry_from_npz_dict(bxrd_options["fileName"]
                                                        +"_boxes.npz", r, 
                                                        allow_pickle=True)[box_ID]   
                for i, iv in enumerate(box_new_float):
                    if box_old[i] == iv:
                        model.stateVarValues = (iv[0] + iv[1])/2.0
                    elif box_old[i][0] != iv[0] and box_old[i][1] != iv[1]:
                        model.stateVarValues = (iv[0] + iv[1])/2.0
                    elif box_old[i][0] == iv[0] and box_old[i][1] != iv[1]: 
                        model.stateVarValues = iv[0] + 0.6667*(iv[1] - iv[0])
                    else:
                        model.stateVarValues = iv[0] + 0.15*(iv[1] - iv[0])
    if "Block" in res_solver.keys():  
        block = res_solver["Block"]
        r = bxrd_options["parent_box_r"] 
        box_ID = bxrd_options["parent_box_ID"] 
        box_old = mostg.get_entry_from_npz_dict(bxrd_options["fileName"]
                                                +"_boxes.npz", r, 
                                                allow_pickle=True)[box_ID]  
        for i in block.colPerm:
            iv = list(block.xBounds_tot[i])
            if list(box_old[i]) == iv:
                block.x_tot[i] = (iv[0] + iv[1])/2.0                  
            elif list(box_old[i])[0] != iv[0] and box_old[i][1] != iv[1]:
                block.x_tot[i] = (iv[0] + iv[1])/2.0
            elif list(box_old[i])[0] == iv[0] and box_old[i][1] != iv[1]: 
                block.x_tot[i] = iv[0] + 0.6667*(iv[1] - iv[0])
            else:
                block.x_tot[i] = iv[0] + 0.15*(iv[1] - iv[0])

def setStateVarValuesToMidPointOfIntervals(res_solver, bxrd_options):
    """ Set all state variable values of a model to the midpoint of their related
    interval
    
    Args:
        :res_solver:     dictionary with resulting model after variable bounds 
                        reduction
        :bxrd_options:   dictionary with user settings used because of absEps. 
                        if mean value is 0, the inital value is set to the absEps
                        value
        
    """
    if "Model" in res_solver.keys():
        model = res_solver["Model"]
    
        if model.xBounds !=[]:
            model.stateVarValues = numpy.empty((len(model.xBounds), 
                                                len(model.xBounds[0])))
            for i in range(len(model.xBounds)):
                for j in range(len(model.xBounds[i])):
                    if isinstance(model.xBounds[i][j], numpy.ndarray):
                        midPoint = (model.xBounds[i][j][0] + 
                                    model.xBounds[i][j][1]) / 2.0
                    else:
                        midPoint = float(mpmath.mpf(model.xBounds[i][j].mid))
                        if midPoint != 0: 
                            model.stateVarValues[i][j]= midPoint
                        else :
                            if (abs(model.xBounds[i][j].a) > 
                                abs(model.xBounds[i][j].b)):
                                model.stateVarValues[i][j] = float(
                                    mpmath.mpf(model.xBounds[i][j].a/2.0))
                            else:
                                model.stateVarValues[i][j] = float(
                                    mpmath.mpf(model.xBounds[i][j].b/2.0))

    if "Block" in res_solver.keys():
        block = res_solver["Block"]
    
        if block.xBounds_tot !=[]:
            #absEps = bxrd_options["absTol"]
            for i in block.colPerm:
                if isinstance(block.xBounds_tot[i], numpy.ndarray):
                    midPoint = (block.xBounds_tot[i][0] + block.xBounds_tot[i][1]) / 2.0
                else:
                    midPoint = float(mpmath.mpf(block.xBounds_tot[i].mid))
                if midPoint != 0: 
                    block.x_tot[i]= midPoint
                else :
                    if abs(block.xBounds_tot[i][0]) > abs(block.xBounds_tot[i][1]):
                        block.x_tot[i] = block.xBounds_tot[i][0]/2.0
                    else:
                        block.x_tot[i] = block.xBounds_tot[i][1]/2.0
