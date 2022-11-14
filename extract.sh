######### https://github.com/SnailTyan/caffe-model-zoo

#### AlexNet --> caffe(ai):
conda activate ai
python AlexNet_GoogleNet_Caffe_Extractor.py -m models/AlexNet/bvlc_alexnet/bvlc_alexnet.caffemodel -n models/AlexNet/bvlc_alexnet/deploy.prototxt -d alexnet


#### GoogleNet --> caffe(ai):
conda activate ai
python AlexNet_GoogleNet_Caffe_Extractor.py -m models/GoogleNet/bvlc_googlenet/bvlc_googlenet.caffemodel -n models/GoogleNet/bvlc_googlenet/deploy.prototxt -d googlenet

#### MobileNet --> tensorflow 1.4(ai2)
python MobileNet_pb_Extractor.py -m models/MobileNet/MobileNet.pb -d mobilenet

#### ResNet50 --> tensorflow 1.4(ai2)
conda activate ai_1.4
python ResNet50_pb_Extractor.py -m models/Resnet50/ResNet50.pb -d resnet50

#### SqueezeNet --> Squeezenet V1.0:
#https://code.ihub.org.cn/projects/271/repository/revisions/master/entry/tools/script/get_model.sh
#https://github.com/forresti/SqueezeNet
conda activate ai
python SqueezeNet_Caffe_Extractor.py -m models/SqueezeNet-master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel -n models/SqueezeNet-master/SqueezeNet_v1.0/deploy.prototxt -d squeezenet



#### AlexNet, GoogleNet and SqueezeNet extraction are same. Just in SqueezeNet you should not create subdir however there is / in names. (Only GoogleNet should create subdirs based on names)

