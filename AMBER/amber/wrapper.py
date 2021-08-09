# -*- coding: utf-8 -*-

"""
Overall wrapper class for AMBER
"""

import tensorflow as tf
from keras import backend as K

try:
    from tensorflow import Session
except ImportError:
    from tensorflow.compat.v1 import Session

    tf.compat.v1.disable_eager_execution()
import os
from . import getter


class Amber:
    """The main wrapper class for AMBER

    This class facilitates the GUI and TUI caller, and should always be maintained

    Parameters
    ----------
    types: dict
    specs: dict

    Attributes
    ----------
    type_dict: dict
    is_built: bool
    model_space: amber.architect.ModelSpace
    controller: amber.architect.BaseController


    Example
    ----------
    PENDING EDITION
    """

    def __init__(self, types, specs=None):
        self.type_dict = types
        self.is_built = False
        self.model_space = None
        self.controller = None
        self.model_fn = None
        self.knowledge_fn = None
        self.reward_fn = None
        self.manager = None
        self.env = None

        # use one tf.Session throughout one DA instance
        self.session = Session()
        try:
            K.set_session(self.session)
        except Exception as e:
            print("Failed to set Keras backend becasue of %s" % e)

        if specs is not None:
            self.from_dict(specs)

    def from_dict(self, d):
        assert type(d) is dict
        print("BUILDING")
        print("-" * 10)
        self.model_space = getter.get_model_space(d['model_space'])
        self.controller = getter.get_controller(controller_type=self.type_dict['controller_type'],
                                                model_space=self.model_space,
                                                session=self.session,
                                                **d['controller'])

        self.model_fn = getter.get_modeler(model_fn_type=self.type_dict['modeler_type'],
                                           model_space=self.model_space,
                                           session=self.session,
                                           controller=self.controller,
                                           **d['model_builder'])

        self.knowledge_fn = getter.get_knowledge_fn(knowledge_fn_type=self.type_dict['knowledge_fn_type'],
                                                    knowledge_data_dict=d['knowledge_fn']['data'],
                                                    **d['knowledge_fn']['params'])

        self.reward_fn = getter.get_reward_fn(reward_fn_type=self.type_dict['reward_fn_type'],
                                              knowledge_fn=self.knowledge_fn,
                                              **d['reward_fn'])

        self.manager = getter.get_manager(manager_type=self.type_dict['manager_type'],
                                          model_fn=self.model_fn,
                                          reward_fn=self.reward_fn,
                                          data_dict=d['manager']['data'],
                                          session=self.session,
                                          **d['manager']['params'])

        self.env = getter.get_train_env(env_type=self.type_dict['env_type'],
                                        controller=self.controller,
                                        manager=self.manager,
                                        **d['train_env'])
        self.is_built = True
        return self

    def run(self):
        assert self.is_built
        self.env.train()
        self.controller.save_weights(os.path.join(self.env.working_dir, "controller_weights.h5"))
