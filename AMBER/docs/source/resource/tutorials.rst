Tutorials
=========

*Last updated: November 16, 2020*

Here we provide a step-by-step guide on how to use AMBER to search Convolutional Neural Network (CNN).
The task is to predict a set of 919 binary labels of epigenetics markers on a 1000-bp DNA sequence.

You can follow the tutorial in real-time using Google Colab. Link to be inserted.

This tutorial assumes you are working on a hpc cluster with Linux environment, with at least 64GB of storage and 16GB
of RAM.

.. Todo::
    Update the dataset to reduced size; i.e. use a toy example instead of the full data.

Setup and Download
-------------------

Please install ``AMBER`` using `our tutorial here <https://amber-dl.readthedocs.io/en/latest/overview/installation.html>`_.

The pre-compiled training/validation/testing data for 919 epigenetics regulatory features can be found here:
http://deepsea.princeton.edu/help/
Be sure to download the `Code + Training data bundle (3.7G)`.


Once downloaded, you can use the following Python code snippet to read the ``*.mat`` files.
This code snippet can also be found in my GitHub repository
`deepsea_keras <https://github.com/zj-zhang/deepsea_keras/blob/master/deepsea_keras/read_data.py>`_.

.. code-block:: python

    import h5py
    from scipy.io import loadmat
    import numpy as np
    import pandas as pd


    def read_train_data(fp=None):
        fp = fp or "./data/train.mat"
        f = h5py.File(fp, "r")
        print(list(f.keys()))
        y = f['traindata'].value
        x = f['trainxdata'].value
        x = np.moveaxis(x, -1, 0)
        y = np.moveaxis(y, -1, 0)
        return x, y


    def read_val_data(fp=None):
        fp = fp or "./data/valid.mat"
        f = loadmat(fp)
        print(list(f.keys()))
        x = f['validxdata']
        y = f['validdata']
        x = np.moveaxis(x, 1, -1)
        return x, y


    def read_test_data(fp=None):
        fp = fp or "./data/test.mat"
        f = loadmat(fp)
        print(list(f.keys()))
        x = f['testxdata']
        y = f['testdata']
        x = np.moveaxis(x, 1, -1)
        return x, y

Next it's time to design the model space.


Design the Model Search Space
------------------------------

We will write a function to render model space that hosts Convolutional Neural Networks. Note that you can freely change
these parameters, such as `kernel_size` and `activation`.

To see a list of available `Operation` arguments, check out here:
https://amber-dl.readthedocs.io/en/latest/amber.architect.html

.. code-block:: python

    def get_model_space(out_filters=64, num_layers=9):
        model_space = ModelSpace()
        num_pool = 4
        expand_layers = [num_layers//4-1, num_layers//4*2-1, num_layers//4*3-1]
        for i in range(num_layers):
            model_space.add_layer(i, [
                Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
                Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
                Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
                Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
                Operation('identity', filters=out_filters),
          ])
            if i in expand_layers:
                out_filters *= 2
        return model_space


Define AMBER components and specifications
------------------------------------------

This is the eseential part of running `AMBER`. First, define the components we need to use.

.. code-block:: python

    type_dict = {
        'controller_type': 'GeneralController',
        'modeler_type': 'EnasCnnModelBuilder',
        'knowledge_fn_type': 'zero',
        'reward_fn_type': 'LossAucReward',
        'manager_type': 'EnasManager',
        'env_type': 'EnasTrainEnv'
    }

There is a growing number of components in `AMBER`. One can easily access them by parsing a string of their names.

Next, some basic information of the training data.

.. code-block:: python

    wd = "./outputs/AmberDeepSea/"
    if os.path.isdir(wd):
      shutil.rmtree(wd)
    os.makedirs(wd)
    input_node = Operation('input', shape=(1000, 4), name="input")
    output_node = Operation('dense', units=919, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
    }

    model_space = get_model_space(out_filters=8, num_layers=6)

Note that the model space has been simplified for running with in Google Colab; again, feel free to tune it as long as
resources permit. Keep in mind that the model space grows exponentially with the number of layers and convolutional kernels.

Finally, we can parse some details about this `AMBER` search.

.. code-block:: python

    specs = {
        'model_space': model_space,

        'controller': {
                'share_embedding': {i:0 for i in range(1, len(model_space))},
                'with_skip_connection': True,
                'skip_target': 0.4,
                'kl_threshold': 0.01,
                'train_pi_iter': 10,
                'buffer_size': 1,
                'batch_size': 20
        },

        'model_builder': {
            'dag_func': 'EnasConv1dDAG',
            'batch_size': 1000,
            'inputs_op': [input_node],
            'outputs_op': [output_node],
            'model_compile_dict': model_compile_dict,
             'dag_kwargs': {
                'stem_config': {
                    'flatten_op': 'flatten',
                    'fc_units': 925
                }
            }
        },

        'knowledge_fn': {'data': None, 'params': {}},

        'reward_fn': {'method': 'auc'},

        'manager': {
            'data': {
                'train_data': train_data,
                'validation_data': val_data
            },
            'params': {
                'epochs': 1,
                'child_batchsize': 512,
                'store_fn': 'minimal',
                'working_dir': wd,
                'verbose': 2
            }
        },

        'train_env': {
            'max_episode': 20,            # has been reduced for running in colab
            'max_step_per_ep': 10,        # has been reduced for running in colab
            'working_dir': wd,
            'time_budget': "00:15:00",    # has been reduced for running in colab
            'with_input_blocks': False,
            'with_skip_connection': True,
            'child_train_steps': 10,      # has been reduced for running in colab
            'child_warm_up_epochs': 1
        }
    }


Run AMBER search and Understand its Outputs
-------------------------------------------

Now construct an instance of `Amber` and hit run.

.. code-block:: python

    # finally, run program
    amb = Amber(types=type_dict, specs=specs)
    amb.run()

This will run till the `time_budget` runs outs, or the `max_episode` reaches, whichever comes first. In this toy example,
we only train `child_train_steps=10` for a maximum of 20 controller steps, with a time limit of 15 minutes. Thus, it should
finish running pretty quickly.

