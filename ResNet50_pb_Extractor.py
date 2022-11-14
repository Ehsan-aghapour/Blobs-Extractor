#!/usr/bin/env python
""" Extract trainable parameters from a frozen model and stores them in numpy arrays.
Usage:
    python tf_frozen_model_extractor -m path_to_frozem_model -d path_to_store_the_parameters

Saves each variable to a {variable_name}.npy binary file.

Note that the script permutes the trainable parameters to NCHW format. This is a pretty manual step thus it's not thoroughly tested.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

#Ehsan
import re

strings_to_remove=["read", "/:0"]
permutations = { 1 : [0], 2 : [1, 0], 3 : [2, 1, 0], 4 : [3, 2, 0, 1]}

alpha_to_number={'a':1,'b':2,'c':3,'d':4,'e':5,'f':6, '_':'shortcut'}

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('Extract TensorFlow net parameters')
    parser.add_argument('-m', dest='modelFile', type=str, required=True, help='Path to TensorFlow frozen graph file (.pb)')
    parser.add_argument('-d', dest='dumpPath', type=str, required=False, default='./', help='Path to store the resulting files.')
    parser.add_argument('--nostore', dest='storeRes', action='store_false', help='Specify if files should not be stored. Used for debugging.')
    parser.set_defaults(storeRes=True)
    args = parser.parse_args()

    # Create directory if not present
    if not os.path.exists(args.dumpPath+'/cnn_data/resnet50_model'):
        os.makedirs(args.dumpPath+'/cnn_data/resnet50_model')

    

    # Extract parameters
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            print("Loading model.")
            with gfile.FastGFile(args.modelFile, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()

                tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="", op_dict=None, producer_op_list=None)

                i=0
                j=0
                for op in graph.get_operations():
                    j=j+1
                    for op_val in op.values():
                        varname = op_val.name
                        #print(f'{j}:{i}: {varname}')
                        i=i+1
                        # Skip non-const values
                        if "read" in varname:
                            t  = op_val.eval()
                            tT = t.transpose(permutations[len(t.shape)])
                            t  = np.ascontiguousarray(tT)

                            for s in strings_to_remove:
                                varname = varname.replace(s, "")
                            if os.path.sep in varname:
                                varname = varname.replace(os.path.sep, '_')
                                print("1-Renaming variable {0} to {1}".format(op_val.name, varname))

                            # Store files
                            if args.storeRes:
                                varname=varname.replace("kernel","weights")
                                varname=varname.replace("fc1000","logits")
                                varname=varname.replace("logits_bias","logits_biases")
                                name=""
                                if varname.startswith("bn"):
                                    numbers=re.findall(r'\d+',varname)
                                    block_number=int(numbers[0])-1
                                    if block_number > 0 :
                                        print(varname)
                                        unit=int(alpha_to_number[varname[3]])
                                        branch=varname.split('branch')[1][0]
                                        prefix=""
                                        if branch=='1':
                                            prefix="shortcut_"
                                        else:
                                            sub_branch=alpha_to_number[ varname.split('branch')[1][1] ]
                                            prefix="conv"+str(sub_branch)+"_"
                                        post=[m.start() for m in re.finditer(r"_",varname)][1]
                                        varname="block"+str(block_number)+"_unit_"+str(unit)+"_bottleneck_v1_"+prefix+"BatchNorm"+varname[post:]
                                    else:
                                        positions=[m.start() for m in re.finditer(r"_",varname)]
                                        varname=varname[positions[0]+1:positions[1]+1]+"BatchNorm"+varname[positions[1]:]
                                
                                if varname.startswith("res"):
                                    numbers=re.findall(r'\d+',varname)
                                    block_number=int(numbers[0])-1                           
                                    unit=int(alpha_to_number[varname[4]])
                                    branch=varname.split('branch')[1][0]
                                    prefix=""
                                    if branch=='1':
                                        prefix="shortcut_"
                                    else:
                                        sub_branch=alpha_to_number[ varname.split('branch')[1][1] ]
                                        prefix="conv"+str(sub_branch)+"_"
                                    post=[m.start() for m in re.finditer(r"_",varname)][1]
                                    varname="block"+str(block_number)+"_unit_"+str(unit)+"_bottleneck_v1_"+prefix+varname[post+1:]
                   
                                
                                #d=re.findall(r'\d+',varname)
                                #if len(d):
                                 #   d=d[0]
                                '''
                                if not ('dw' in varname) and not ('pw' in varname) and len(d):
                                    varname=varname.replace('conv'+d[0],'Conv2d_0').replace('kernel','weights').replace('bn','BatchNorm')
                                    print(f'varname: {varname}')
                                else:
                                    if len(d):
                                        varname=varname.replace('_'+d[0],'').replace('conv','Conv2d_'+d[0]).replace('_bn_','_BatchNorm_')
                                        varname=varname.replace('_dw_','_depthwise_').replace('_pw_','_pointwise_').replace('kernel','weights')
                                        print(f'varnamew: {varname}')

                                if varname == 'conv_preds_kernel':
                                    varname='Logits_Conv2d_1c_1x1_weights'
                                if varname == 'conv_preds_bias':
                                    varname='Logits_Conv2d_1c_1x1_biases'
				'''
                                print("Renaming to {0}".format(varname))    	
                                print("Saving variable {0} with shape {1} ...".format(varname, t.shape))
                                varname='cnn_data/resnet50_model/'+varname
                                np.save(os.path.join(args.dumpPath, varname), t)
