#from common import find_mxnet
#from common.util import get_gpus
import mxnet as mx 
import mxnet.gluon.model_zoo.vision as models 
from importlib import import_module

import argparse,time
import numpy as np
import mxnet as mx


from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn

#from gluoncv.model_zoo import get_model
#from gluoncv.utils import makedirt,TrainingHistory


def main():    
  image_shape = (3,299,299)
  num_layers=50
  network='resnet'
  dtype = 'float32'
  num_batches=100
  batch_size=64
  dev=mx.gpu(0)

  # get the model
  net = import_module('symbols.' + network)
  sym = net.get_symbol(num_classes=1000,image_shape=','.join([str(i) for i in image_shape]),num_layers=num_layers, dtype=dtype)

  data_shape=[('data', (batch_size,)+image_shape)]

  mod = mx.mod.Module(symbol=sym,context=dev)
  mod.bind(for_training=False,inputs_need_grad=False,data_shapes=data_shape) 

  # get fake data
  data = [mx.random.uniform(-1.0,1.0,shape=shape,ctx=dev) for _,shape in mod.data_shapes]
  DataBatch = mx.io.DataBatch(data,[]) # empty labels
  
  #training loop
  train(mod,DataBatch,num_batches,batch_size)


def train(mod,DataBatch,num_batches,batch_size):
  dry_run = 5
  for i in range(dry_run+num_batches):
      if i == dry_run:
          tic = time.time()
      mod.forward(DataBatch,is_train=False)
      for output in mod.get_outputs():
          output.wait_to_read()
  print('batch number #%d img/sec=%f' %(i,(num_batches*batch_size/time.time()-tic)))

if __name__ == '__main__':
  main()

