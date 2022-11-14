import numpy as np
import caffe2
import os

from caffe2.python import (
    brew,
    model_helper,
    workspace,
)

ArmCL_mdl_dir={
'mobilenet_v2':'mobilenet_v2_1.0_224_model',
'bvlc_googlenet':'googlenet_model',
'bvlc_alexnet':'alexnet_model',
'resnet50':'resnet50_model',
'squeezenet':'squeezenet_v1.0_model'
}

image_size={
'bvlc_googlenet':[224,224],
'bvlc_alexnet':[227,227],
'squeezenet':[227,227],
'resnet50':[224,224],
'mobilenet_v2':[224,224]
}

imports = ['bvlc_alexnet', 'bvlc_googlenet', 'resnet50', 'squeezenet','mobilenet_v2']
#imports = ['resnet50']
models = {}
for x in imports:
	print(f"Importing model {x} from caffe2")
	try:
        	models[x]=__import__(f'caffe2.python.models.{x}', fromlist=[''])
        	print ("Successfully imported ", x, '.')
	except ImportError:
		print ("Error importing ", x, '.')
		if os.system(f"python -m caffe2.python.models.download -i {x}") is 0:
			print(f'Model {x} Downloaded successfully')
                        #/anaconda3/envs/ai_tf_1.4/lib/python3.7/site-packages/caffe2/python/models
			models[x]=__import__(f'caffe2.python.models.{x}', fromlist=[''])
			print ("Successfully imported ", x, '.')
			
		else:
			print(f'Error in downloading model {x}. skipping it.')
		
		

print('Models are imported. Extractig data and images...')

for m in imports:
	print(f'Extracting parameters of model {m} ...')
	blobs={}
	workspace.ResetWorkspace()
	workspace.RunNetOnce(models[m].init_net)
	for b in workspace.Blobs():
		blobs[b]=workspace.FetchBlob(b)

	dr=ArmCL_mdl_dir[m]
	dr=f'assets_{m}/cnn_data/'+dr+'/'
	if not os.path.exists(dr):
		os.makedirs(dr)
	else:
		print(f'assets_{m} existed!')
		continue	
				
		
	
	for i,name in enumerate(blobs):
		blob_dr=name[:name.rfind('/')+1]
		blob_name=name.replace('/','_')
		if blob_dr and not os.path.exists(dr+blob_dr):
			os.makedirs(dr+blob_dr)
		#np.save(dr+blob_dr+blob_name,blobs[name])
		print (name)
		print (blobs[name].shape)
		np.save(dr+blob_name,blobs[name])

	print(f'Parameters of model {x} was extracted to {dr}, preparing its images')

	#print('Preparing images...')		
	cmd=f'python3 convert_images.py {m} {image_size[m][0]} {image_size[m][1]}'

	ok=os.system(cmd)
	#print(f'Ok is {ok} type:{type(ok)}')
	while ok is not 0 :
		i=input(f"Add image size for {m} model in convert_images.py and press enter. Press E to exit")
		if i is "E":
			break
		ok=os.system(cmd)
	else:
		print(f'Images added.\nAssets preparation for model {m} finished.')



'''
with open("init_net.pb") as f:
     init_net = f.read()
with open("predict_net.pb") as f:
     predict_net = f.read()        

'''


