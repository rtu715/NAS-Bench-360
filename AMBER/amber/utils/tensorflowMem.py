import os

import tensorflow as tf

num_threads = os.environ.get('OMP_NUM_THREADS')


def get_session(gpu_fraction=0.75):
    """Assume that you have 6GB of GPU memory and want to allocate ~2GB"""

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        print("with nthreads=%s" % num_threads)
        return tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        # return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1}))
        return tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options))


# KTF.set_session(get_session())


def get_session2(CPU, GPU):
    if GPU:
        num_GPU = 1
        num_CPU = 1
    if CPU:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_threads,
                            inter_op_parallelism_threads=num_threads,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU}
                            )

    session = tf.Session(config=config)
    K.set_session(session)
