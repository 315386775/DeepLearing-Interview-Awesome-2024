# -*- coding: utf-8 -*-

import mxnet as mx
import numpy as np
from skimage import io,transform

prefix = 'test2'
epoch = 2000
model = mx.model.FeedForward.load(prefix,epoch,ctx=mx.gpu(),numpy_batch_size=1)
synset = [l.strip() for l in open('synset.txt').readlines()]

mean_img = mx.nd.load('mean.bin').values()[0].asnumpy()

test_img = 'car1.jpg'
img = io.imread(test_img)
io.imshow(img)

short_edge = min(img.shape[:2])
yy = int((img.shape[0] - short_edge)/2)
xx = int((img.shape[1] - short_edge)/2)
crop_img = img[yy:yy + short_edge,xx:xx + short_edge]

resized_img = transform.resize(crop_img,(224,224))
sample = np.asarray(resized_img)*256
sample = np.swapaxes(sample,0,2)
sample = np.swapaxes(sample,1,2)
normed_img = sample - mean_img
normed_img.resize(1,3,224,224)

batch = normed_img
prob = model.predict(batch)[0]
pred = np.argsort(prob)[::-1]

top1 = pred[0]
print(test_img," Top1: ", synset[top1])
print prob