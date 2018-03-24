import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
#%matplotlib inline


#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import sys
import os

import caffe

caffe.set_mode_cpu()

model_def = "deploy.prototxt"
model_weights = "pre_trained_models/EmotiW_VGG_S.caffemodel"

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( 'mean.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
arr= arr[0].mean(1).mean(1)
arr.shape # check the shape of arr
for x in  zip('BGR', arr):

    print ('mean-subtracted value :', x)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', arr)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

image = caffe.io.load_image('frame_det_000128.bmp')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

labels_file = 'classes.txt'

labels = np.loadtxt(labels_file, str, delimiter='\t')

print('predicted class is:', output_prob)

top_inds = output_prob.argsort()[::-1][:7]  # reverse sort and take five largest items

print ('output label:', labels[output_prob.argmax()])

print ('probabilities and labels:')

for x in zip(output_prob[top_inds], labels[top_inds]):
    print(x)
