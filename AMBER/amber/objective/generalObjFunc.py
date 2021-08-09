#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
ZZJ
Nov. 16, 2018
"""

from .generalObjMath import *


class GeneralKnowledgeObjectiveFunction(object):
    """Scaffold of objective knowledge-functions
    """

    def __init__(self):
        self.W_model = None
        self.W_knowledge = None
        self._build_obj_func()

    def __call__(self, model, data, **kwargs):
        self.model_encoder(model, data, **kwargs)
        return self.obj_fn(self.W_model, self.W_knowledge, **kwargs)

    def __str__(self):
        return "Objective function for augmenting neural architecture search"

    def model_encoder(self, model, data, **kwargs):
        """encode $\hat{W}$ from model
        """
        raise NotImplementedError("abstract method")

    def knowledge_encoder(self, **kwargs):
        """encode $\tilde{W}$ from existing knowledge
        """
        raise NotImplementedError("abstract method")

    def _build_obj_func(self, **kwargs):
        def obj_fn(W_model, W_knowledge, **kwargs):
            return None

        self.obj_fn = obj_fn

    def get_obj_val(self, **kwargs):
        return self.obj_fn(self.W_model, self.W_knowledge, **kwargs)


class AuxilaryAcc(GeneralKnowledgeObjectiveFunction):
    """Perform the accuracy measurement (defined by method argument)  on an 
    auxilary dataset
    """
    def __init__(self, method='auc', *args, **kwargs):
        if method == 'auc' or method == 'auroc':
            from sklearn.metrics import roc_auc_score
            self.scorer = roc_auc_score
        elif method == 'aupr' or method == 'auprc':
            from sklearn.metrics import average_precision_score
            self.scorer = average_precision_score
        elif callable(method):
            self.scorer = method
        else:
            raise Exception("cannot understand scorer method: %s" % method)
        super().__init__(*args, **kwargs)

    def model_encoder(self, model, *args, **kwargs):
        self.W_model = model.predict(self.W_knowledge[0])

    def knowledge_encoder(self, data, *args, **kwargs):
        self.W_knowledge = data

    def _build_obj_func(self, **kwargs):
        def obj_fn(W_model, W_knowledge, **kwargs):
            y = W_knowledge[1]
            pred = W_model
            auc_list = []
            if type(y) is not list:
                y = [y]
            if type(pred) is not list:
                pred = [pred]
            for i in range(len(y)):
                tmp = []
                if len(y[i].shape) == 1: y[i] = np.expand_dims(y[i], axis=-1)
                if len(pred[i].shape) == 1: pred[i] = np.expand_dims(pred[i], axis=-1)
                for j in range(y[i].shape[1]):
                    try:
                        score = self.scorer(y_true=y[i][:, j], y_score=pred[i][:, j])
                        tmp.append(score)
                    except ValueError:  # only one-class present
                        pass
                auc_list.append(tmp)
            avg_auc = np.nanmean(np.concatenate(auc_list, axis=0))
            return avg_auc
        self.obj_fn = obj_fn
