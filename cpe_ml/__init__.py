import tensorflow as tf
import numpy as np
import time
import keras

from keras.callbacks import Callback
from keras import backend as K
import ml_comm as mc
import math

#init and finalize
def init():
    mc.init_mpi()
    
def finalize():
    mc.finalize()
    
#some mpicomm features
def size():
    return mc.get_nranks()
    
def rank():
    return mc.get_rank()
