from common import find_mxnet
from common.util import get_gpus
import mxnet as mx
import mxnet.gloun.module_zoo.vision as vision
from importlib import import_module 
import logging
import argparse
import time
import numpy as np

def main():

  # load data set - set fake data
  input_labels = np.rand(5,5,5)
  targets = np.rand(5,5,5)

  # get model 
  model_resnet50 = vision.resnet50_v1(pretrained=false)
  
  # pick optimizer 
  optim = mx.optimizer.SGD();
  
  # make a data loader object 

  # run training
  train(model,input_labels,targets,optim)

def train(model,data,targets,optim):
  # set an optimizer


  




