import ctypes as ct
import numpy as np
import os
import platform

class BeeDNNLoader:
    c_float_p = ct.POINTER(ct.c_float)
    
    # Load library
    lib_name = "BeeDNNLib"
    if platform.system() == "Windows":
        lib_name += ".dll"
    elif platform.system() == "Darwin":
        lib_name += ".dylib"
    else:
        lib_name += ".so"
        
    # Try to load from current directory or build directory
    try:
        if os.path.exists("./" + lib_name):
            lib = ct.cdll.LoadLibrary("./" + lib_name)
        elif os.path.exists("../python_binding/" + lib_name): # Dev path
             lib = ct.cdll.LoadLibrary("../python_binding/" + lib_name)
        else:
             lib = ct.cdll.LoadLibrary(lib_name)
    except Exception as e:
        print(f"Failed to load {lib_name}: {e}")
        raise e

    # Define function signatures
    lib.create_net.restype = ct.c_void_p
    lib.delete_net.argtypes = [ct.c_void_p]
    
    lib.init_net.argtypes = [ct.c_void_p, ct.c_int32]
    
    # Net methods
    lib.add_layer.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.set_classification_mode.argtypes = [ct.c_void_p, ct.c_int32]
    lib.set_train_mode.argtypes = [ct.c_void_p, ct.c_int32]
    lib.predict.argtypes = [ct.c_void_p, c_float_p, c_float_p, ct.c_int32, ct.c_int32]
    
    # Layer methods
    lib.layer_create_activation.argtypes = [ct.c_char_p]
    lib.layer_create_activation.restype = ct.c_void_p
    
    lib.layer_construct.argtypes = [ct.c_char_p, c_float_p, ct.c_int32, ct.c_char_p]
    lib.layer_construct.restype = ct.c_void_p
    
    lib.delete_layer.argtypes = [ct.c_void_p]
    
    # NetTrain methods
    lib.create_train.restype = ct.c_void_p
    lib.delete_train.argtypes = [ct.c_void_p]
    
    lib.train_set_train_data.argtypes = [ct.c_void_p, c_float_p, ct.c_int, ct.c_int, c_float_p, ct.c_int, ct.c_int]
    lib.train_set_validation_data.argtypes = [ct.c_void_p, c_float_p, ct.c_int, ct.c_int, c_float_p, ct.c_int, ct.c_int]
    lib.train_set_batch_size.argtypes = [ct.c_void_p, ct.c_int]
    lib.train_set_epochs.argtypes = [ct.c_void_p, ct.c_int]
    lib.train_fit.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.train_set_optimizer.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.train_set_loss.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.train_set_regularizer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_float]
    
    lib.train_get_train_loss_size.argtypes = [ct.c_void_p]
    lib.train_get_train_loss_size.restype = ct.c_int
    lib.train_get_train_loss_data.argtypes = [ct.c_void_p, c_float_p]

    def __init__(self, inputSize=0):
        self.net = ct.c_void_p(self.lib.create_net())
        if inputSize > 0:
            self.lib.init_net(self.net, inputSize)
        self.inputSize = inputSize
        self.trainer = None

    def __del__(self):
        if self.lib:
            self.lib.delete_net(self.net)
            if self.trainer:
                self.lib.delete_train(self.trainer)

    def add_layer(self, layer_name, args=[], arg_str=""):
        # Check if it's a simple activation (compat)
        # Using layer_construct for everything is safer if we can.
        # But 'layer_name' in old code likely meant just "Dense" or "Tanh".
        # If args provided, use construct.
        if args or (arg_str and len(arg_str)>0) or layer_name not in ["Sigmoid", "Tanh", "ReLU", "LeakyReLU", "Elu", "Softmax"]:
            c_type = ct.c_char_p(layer_name.encode('utf-8'))
            c_arg_str = ct.c_char_p(arg_str.encode('utf-8'))
            c_args = (ct.c_float * len(args))(*args)
            layer = self.lib.layer_construct(c_type, c_args, len(args), c_arg_str)
            self.lib.add_layer(self.net, layer)
        else:
             # Try activation
            c_activ = ct.c_char_p(layer_name.encode('utf-8'))
            layer = self.lib.layer_create_activation(c_activ)
            self.lib.add_layer(self.net, layer)

    def set_classification_mode(self, bClassificationMode):
        self.lib.set_classification_mode(self.net, ct.c_int32(bClassificationMode))
        
    def set_train_mode(self, bTrainMode):
        self.lib.set_train_mode(self.net, ct.c_int32(bTrainMode))
 
    def predict(self, mIn, mOut=None):
        data_in = mIn.astype(np.float32)
        nbSamples = mIn.shape[0]
        nbFeatures = mIn.shape[1]
        
        if mOut is None:
            # We need to know output size. 
            # Not exposed easily.
            # Assuming user passes mOut or we guess?
            # Old code: mOut passed.
            raise ValueError("mOut argument required")

        data_p_in = data_in.ctypes.data_as(self.c_float_p)
        data_p_out = mOut.ctypes.data_as(self.c_float_p)
 
        self.lib.predict(self.net, data_p_in, data_p_out, nbSamples, nbFeatures)

    # Trainer API
    def create_trainer(self):
        self.trainer = ct.c_void_p(self.lib.create_train())
        return self.trainer

    def fit(self, samples, truth, batch_size=32, epochs=100, optimizer="Adam", loss="MeanSquareError", lr=0.01):
        if not self.trainer: self.create_trainer()
        
        s_data = samples.astype(np.float32)
        t_data = truth.astype(np.float32)
        
        self.lib.train_set_train_data(self.trainer, 
            s_data.ctypes.data_as(self.c_float_p), s_data.shape[0], s_data.shape[1],
            t_data.ctypes.data_as(self.c_float_p), t_data.shape[0], t_data.shape[1])
            
        self.lib.train_set_batch_size(self.trainer, batch_size)
        self.lib.train_set_epochs(self.trainer, epochs)
        self.lib.train_set_optimizer(self.trainer, optimizer.encode('utf-8'))
        self.lib.train_set_loss(self.trainer, loss.encode('utf-8'))
        
        # Train
        self.lib.train_fit(self.trainer, self.net)
        
    def get_train_loss(self):
        if not self.trainer: return []
        size = self.lib.train_get_train_loss_size(self.trainer)
        data = np.zeros(size, dtype=np.float32)
        self.lib.train_get_train_loss_data(self.trainer, data.ctypes.data_as(self.c_float_p))
        return data
