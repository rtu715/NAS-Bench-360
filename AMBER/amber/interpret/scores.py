import numpy as np


class PrecisionAtRecall:
    def __init__(self, cut_points, *args, **kwargs):
        from sklearn.metrics import precision_recall_curve
        self.scorer = precision_recall_curve
        self.cut_points = cut_points
        self.pred = kwargs.pop('pred', None)
        self.cut_index = 1
        self.eval_index = 0
        self.compare_op = kwargs.pop('compare_op', 'max')

    def __call__(self, model, data, *args, **kwargs):
        MIN_POS = 50  # at least this number to get a stable estimate
        X, y = data
        if self.pred is None:
            pred = model.predict(X)
        else:
            pred = self.pred
        if type(y) is not list:
            y = [y]
        if type(pred) is not list:
            pred = [pred]
        auc_list = []
        for i in range(len(y)):
            tmp = []
            for j in range(y[i].shape[1]):
                if np.sum(y[i][:, j]) < MIN_POS:
                    this = {x: np.nan for x in self.cut_points}
                else:
                    curve = self.scorer(y[i][:, j], pred[i][:, j])
                    this = {}
                    for cutpoint in self.cut_points:
                        if self.compare_op == 'max':
                            idx = np.max(np.where(curve[self.cut_index] > cutpoint))
                        elif self.compare_op == 'min':
                            idx = np.min(np.where(curve[self.cut_index] > cutpoint))
                        else:
                            raise Exception("compare_op must be `max` or `min`; got %s" % self.compare_op)
                        this[cutpoint] = curve[self.eval_index][idx]
                tmp.append(this)
            auc_list.append(tmp)
        self.auc_list = auc_list  # auc_list: [ [this0_0, this0_1, this0_2, ..], [this1_0, this1_1, ..] ]
        return auc_list


class TprAtFpr(PrecisionAtRecall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.metrics import roc_curve
        self.scorer = roc_curve
        self.cut_index = 0
        self.eval_index = 1
