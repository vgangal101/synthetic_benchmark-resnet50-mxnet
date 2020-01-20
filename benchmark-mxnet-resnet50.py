from common import find_mxnet
from common.util import get_gpus
import mxnet as mx
import mxnet.gloun.module_zoo.vision as vision
from gluoncn.model_zoo import get_model
from importlib import import_module 
import logging
import argparse
import time
import numpy as np

def main():
  batch_size=64
  num_classes=1000
  epoch_size=100 
  num_epochs=1
  image_shape=(3,229,229)
  # epoch_size is similar to the idea of steps

  # set fake data
  network='resnet'
  num_layers=50
  dev = mx.gpu(0) if len(get_gpus()) > 0 else mx.cpu()

  net= import_module('symbols.'+network)
  sym= net.get_symbol(num_classes=num_classes,image_shape=image_shape,num_layers=num_layers,dtype=np.float32)
  mod = mx.mod.Module(symbol=sym,context=dev)
  data = [mx.random.uniform(-1.0,1.0,shape=shape,ctx=dev) for _, shape in mod.data_shapes]
  DataIter = mx.io.DataBatch(data,[])

  # get model 
  model_resnet50 = vision.resnet50_v1(pretrained=false)

  # pick optimizer 
  optim = mx.optimizer.SGD();
  
  # run training
  train(model_resnet50,DataIter,optim)

def train(model,DataIter,optim):
  for  
  

  




