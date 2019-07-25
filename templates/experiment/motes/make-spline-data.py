#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:38:40 2018

@author: debus
"""

import glob, os, pickle, sys
import numpy as np

output_file = 'spline-data.h'


header = \
'''#ifndef SPLINE_DATA_H_
#define SPLINE_DATA_H_

#include "spline-detector.h"

#define MAX_NEIGHBORS ###NUM_FILES###

'''


footer = '''

#endif /* SPLINE_DATA_H_ */
'''


single_struct_template = \
'''
static struct spline_data ###NAME### = {
    .spline_coefficients = ###COEFFICIENTS###,
    .bandwidth = ###BANDWIDTH###,
    .right_boundary = ###RIGHT_BOUNDARY###
};

'''


structs_array_template = \
'''
    { // ###NAME###
    .spline_coefficients = ###COEFFICIENTS###,
    .bandwidth = ###BANDWIDTH###,
    .right_boundary = ###RIGHT_BOUNDARY###
    }'''


select_template = '''
    
    
struct spline_data* select_spline_set(unsigned num_neighbors)
{
     switch(num_neighbors)
     {
###CASES###
        default: return spline_###NUM_FILES###_neighbor;
     }
     return spline_###NUM_FILES###_neighbor;
}
'''


def make_single_structs(pickle_dict):
    generated_file = header
    
    for i in sorted(pickle_dict.keys()):
        generated_file += "\n\n /* Spline data for %s neighbors */ \n" %i
        spline_dict = pickle_dict[i]['spline']
        kde_dict = pickle_dict[i]['kde']
        
        for feature_name in sorted(spline_dict.keys()):
            spline_data =   spline_dict[feature_name]  

            struct_name = ("spline_%s_" %i) + feature_name.replace("* ","").replace("/","").replace(" ","_").lower()
            right_boundary = str(spline_data.x[-1])
            bandwidth = str(kde_dict[feature_name].bandwidth)
            c_matrix = np.array(spline_data.c).T
            coefficients = "{" + ", ".join(["{" + ", ".join([str(x) for x in row]) + "}" for row in c_matrix]) + "}"
            
            struct_code = single_struct_template.replace("###NAME###", struct_name)
            struct_code = struct_code.replace("###COEFFICIENTS###", coefficients)
            struct_code = struct_code.replace("###BANDWIDTH###", bandwidth)
            struct_code = struct_code.replace("###RIGHT_BOUNDARY###", right_boundary)
                                              
            generated_file += struct_code
    
    generated_file += footer

    return generated_file    


def make_structs_array(pickle_dict):
    generated_file = header.replace("###NUM_FILES###", str(len(pickle_dict)-1))
    
    array_dict = {}
    for i in sorted(pickle_dict.keys()):
        struct_name =  "spline_%s_neighbor" %i
        array_dict[i] = struct_name
               
        spline_dict = pickle_dict[i]['spline']
        kde_dict = pickle_dict[i]['kde']

        generated_file += "\n\n /* Spline data for %s neighbors */ \n" %i
        generated_file += "static struct spline_data %s[%d] = {\n" %(struct_name, len(spline_dict))
                
        structs = []
               
        for feature_name in sorted(spline_dict.keys()):
            spline_data =   spline_dict[feature_name]  

            right_boundary = str(spline_data.x[-1])
            bandwidth = str(kde_dict[feature_name].bandwidth)
            c_matrix = np.array(spline_data.c).T
            coefficients = "{" + ", ".join(["{" + ", ".join([str(x) for x in row]) + "}" for row in c_matrix]) + "}"
            
            struct_code = structs_array_template.replace("###NAME###", feature_name)
            struct_code = struct_code.replace("###COEFFICIENTS###", coefficients)
            struct_code = struct_code.replace("###BANDWIDTH###", bandwidth)
            struct_code = struct_code.replace("###RIGHT_BOUNDARY###", right_boundary)
                                              
            structs.append(struct_code)
            
        generated_file +=  ",\n".join(structs)
        generated_file += "\n};"
    

    cases = "\n".join(["\t\tcase %s: return %s;" %(k, array_dict[k]) for k in sorted(array_dict.keys())]) 
    generated_file += select_template.replace("###CASES###",cases).replace("###NUM_FILES###", str(len(pickle_dict)-1))
    
    generated_file += footer

    return generated_file    


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("EXITING, no arguments!")
        sys.exit(-1)
    
    data_folder = sys.argv[1]
    files = glob.glob(os.path.join(data_folder, "*.pkl"))
    
    pickle_dict = {}
    
    for f in files:
        num_neighbors = os.path.splitext(os.path.basename(f))[0]
        with open(f, 'rb') as fd:
            pickles = pickle.load(fd, encoding='utf8')

        pickle_dict[num_neighbors] = pickles
    
    generated_file = make_structs_array(pickle_dict)
    
    with open(output_file, "w") as f:
        f.write(generated_file)
    print("Writing output to: " + os.path.abspath(output_file))
            
        
    