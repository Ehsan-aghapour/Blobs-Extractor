#!/usr/bin/env python
"""Extracts trainable parameters from Caffe models and stores them in numpy arrays.
Usage
    python caffe_data_extractor -m path_to_caffe_model_file -n path_to_caffe_netlist

Saves each variable to a {variable_name}.npy binary file.

Tested with Caffe 1.0 on Python 2.7
"""
import argparse
import caffe
import os
import numpy as np


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('Extract Caffe net parameters')
    parser.add_argument('-m', dest='modelFile', type=str, required=True, help='Path to Caffe model file')
    parser.add_argument('-n', dest='netFile', type=str, required=True, help='Path to Caffe netlist')

    #Ehsan
    parser.add_argument('-d', dest='destination', type=str, required=True, help='path to destination')

    args = parser.parse_args()

    # Create Caffe Net
    net = caffe.Net(args.netFile, 1, weights=args.modelFile)

    #Ehsan
    dist=args.destination
    if(dist[-1]!='/'):
        dist=dist+'/'
    os.makedirs(dist)
    print(dist)
    input()

    # Read and dump blobs
    for name, blobs in net.params.items():
        print('Name: {0}, Blobs: {1}'.format(name, len(blobs)))
        for i in range(len(blobs)):
            # Weights
            if i == 0:
                outname = name + "_w"
            # Bias
            elif i == 1:
                outname = name + "_b"
            else:
                continue

            varname = outname
            if os.path.sep in varname:
                varname2 = varname.replace(os.path.sep, '_')
                varname3 = varname[:varname.find('/')+1]
                varname4 = varname3 + varname2
                if not os.path.exists(dist+varname3):
                    print (f'creating {dist}{varname3}')
                    os.makedirs(dist+varname3)
                print("Renaming variable {0} to {1}".format(outname, varname4))
            print("Saving variable {0} with shape {1} ...".format(varname4, blobs[i].data.shape))
            # Dump as binary
            np.save(dist+varname4, blobs[i].data)
