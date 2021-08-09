import tensorflow as corrected_tf
if corrected_tf.__version__.startswith("2"):
    corrected_tf.compat.v1.disable_eager_execution()
    import tensorflow.compat.v1 as corrected_tf


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']





