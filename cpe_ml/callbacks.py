import tensorflow as tf
import numpy as np
import time
import keras

from keras.callbacks import Callback
from keras import backend as K
import ml_comm as mc
import math


class _DistributedOptimizer(keras.optimizers.Optimizer):
    """
    Leveraging approach used in horovod.keras.DistributedOptimizer.
    """

    def __init__(self, name, **kwargs):
        if name is None:
            name = "Distributed%s" % self.__class__.__base__.__name__
        self._name = name
        super(self.__class__, self).__init__(**kwargs)

    def get_gradients(self, loss, params):
        grads = super(self.__class__, self).get_gradients(loss, params)
        grads_mc = mc.gradients(grads, 0)
        return grads_mc

def DistributedOptimizer(optimizer, name=None):
    """
    An optimizer that wraps another keras.optimizers.Optimizer
    """
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(name, **optimizer.get_config())


class InitPluginCallback(Callback):
    def __init__(self, max_steps, buffer_size):
        import ml_comm as mc
        super(InitPluginCallback, self).__init__()
        self.max_steps = max_steps
        self.buffer_size = buffer_size
    def on_train_begin(self, logs=None):
        mc.init(1, 1, self.buffer_size, "tensorflow")
        mc.config_team(0, 0, int(0.2*self.max_steps), self.max_steps, 2, 100)

class BroadcastVariablesCallback(Callback):
    def __init__(self, head_rank, validate=False):
        import ml_comm as mc
        super(BroadcastVariablesCallback, self).__init__()
        self.head_rank = head_rank
        self.validate  = validate
    def on_train_begin(self, logs=None):
        sess = K.get_session()
        
        # Split variables based on type -> float32 vs all else
        test_v = tf.Variable([0], dtype=tf.float32)
        all_vars = tf.trainable_variables()
        float_vars = [v for v in all_vars if v.dtype == test_v.dtype]
        other_vars = [v for v in all_vars if v.dtype != test_v.dtype]

        # Initialize variables and broadcast from head node
        sess.run(tf.variables_initializer(all_vars))
        new_vars = mc.broadcast(float_vars, 0)
        bcast = tf.group(*[tf.assign(v, new_vars[k]) for k,v in enumerate(float_vars)])
        sess.run(bcast)

        # Validate Broadcast
        if self.validate:
            py_all_vars = [sess.run(v) for v in float_vars]
            var_types = [np.array([v]) if type(v) == np.float32 else v for v in py_all_vars]
            if mc.get_rank() is 0:
                if (mc.check_buffers_match(var_types, 1) != 0):
                    tf.logging.error("Not all processes have the same initial model!")
                else:
                    tf.logging.info("Initial model is consistent on all ranks")


class _DistributedOptimizer(keras.optimizers.Optimizer):
    """
    Leveraging approach used in horovod.keras.DistributedOptimizer.
    """

    def __init__(self, name, **kwargs):
        if name is None:
            name = "Distributed%s" % self.__class__.__base__.__name__
        self._name = name
        super(self.__class__, self).__init__(**kwargs)

    def get_gradients(self, loss, params):
        grads = super(self.__class__, self).get_gradients(loss, params)
        grads_mc = mc.gradients(grads, 0)
        return grads_mc

def DistributedOptimizer(optimizer, name=None):
    """
    An optimizer that wraps another keras.optimizers.Optimizer
    """
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(name, **optimizer.get_config())