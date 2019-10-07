import mxnet as mx
import numpy as np
import os
import time


def convert_gluon_to_symbolic(
    model, remove_file_after=True, fake_data_shape=[1, 3, 224, 224]
):
    """convert_gluon_to_symbolic [summary]
    
    Parameters
    ----------
    model : gluon model
        [description]
    remove_file_after : bool, optional
        remove file generated in process of conversion, by default True
    fake_data_shape : list, optional
        shape of data to be fed to your network, by default [1, 3, 224, 224]
    
    Returns
    -------
    Symbol, dict, dict
    sym, arg, aux
    """

    model.hybridize()
    fake_data = np.zeros(fake_data_shape)
    fake_data = mx.nd.array(fake_data)
    model.forward(fake_data)
    current_time = time.time()
    model.export("exported_model" + str(current_time))
    sym, arg, aux = mx.model.load_checkpoint("exported_model" + str(current_time), 0)
    if remove_file_after:
        try:
            os.remove("exported_model" + str(current_time) + "-0000.params")
            os.remove("exported_model" + str(current_time) + "-symbol.json")
        except OSError as e:
            print(e)

    return sym, arg, aux
