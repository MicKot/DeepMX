import mxnet as mx
import numpy as np
import os


def convert_gluon_to_symbolic(
    model,
    remove_tmp_file=True,
    input_shape=[1, 3, 224, 224],
    model_name="tmp_exported_model",
):
    """convert_gluon_to_symbolic [summary]
    
    Parameters
    ----------
    model : gluon model
        [description]
    remove_tmp_file : bool, optional
        remove file generated in process of conversion, by default True
    input_shape : list, optional
        shape of input data, by default [1, 3, 224, 224]
    
    Returns
    -------
    Symbol, dict, dict
    sym, arg, aux
    """

    model.hybridize()
    fake_data = np.zeros(input_shape)
    fake_data = mx.nd.array(fake_data)
    model.forward(fake_data)
    model.export(model_name)
    sym, arg, aux = mx.model.load_checkpoint(model_name, 0)
    if remove_tmp_file:
        try:
            os.remove(model_name + "-0000.params")
            os.remove(model_name + "-symbol.json")
        except OSError as e:
            print(e)

    return sym, arg, aux
