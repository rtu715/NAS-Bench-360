# -*- coding: UTF-8 -*-

from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from .generalObjFunc import GeneralKnowledgeObjectiveFunction
from ..architect.commonOps import batchify

from .motifUtils import group_motif_by_factor, compare_motif_diff_size, get_seq_chunks, create_conv_from_motif, remove_dup_motif


class MotifKLDivergence(GeneralKnowledgeObjectiveFunction):
    def __init__(self, temperature, Lambda_regularizer, is_multiGPU_model=False):
        super(MotifKLDivergence, self).__init__()
        self.temperature = temperature
        self.Lambda_regularizer = Lambda_regularizer
        self.is_multiGPU_model = is_multiGPU_model

    def __call__(self, model, data, **kwargs):
        """Motif_obj_function is independent of the data
        """
        self.model_encoder(model, None)
        return self.obj_fn(self.W_model, self.W_knowledge, self.Lambda_regularizer)

    def __str__(self):
        return 'Motif K-function for Interpretable Model Learning'

    def model_encoder(self, model, data, **kwargs):
        if self.is_multiGPU_model:
            multigpu_models = [model.layers[i] for i in range(len(model.layers)) if
                               model.layers[i].name.startswith('model')]
            assert len(multigpu_models) == 1
            layer_dict = {multigpu_models[0].layers[i].name: multigpu_models[0].layers[i] for i in
                          range(len(multigpu_models[0].layers))}
        else:
            layer_dict = {model.layers[i].name: model.layers[i] for i in range(len(model.layers))}
        W = layer_dict['conv1'].get_weights()[0]
        # W dimenstion: filter_len, num_channel(=4), num_filters for Conv1D
        # for Conv2D: num_channel/filter_height(=4), filter_len/filter_len, 1, num_filters
        if len(W.shape) == 4:
            W = np.squeeze(W, axis=2)
            W = np.moveaxis(W, [0, 1], [1, 0])
        # either way, num_filters is the last dim
        num_filters = W.shape[-1]
        W_prob = np.zeros((W.shape[2], W.shape[0], W.shape[1]))
        beta = 1. / self.temperature
        for i in range(num_filters):
            w = W[:, :, i].copy()
            for j in range(w.shape[0]):
                w[j, :] = np.exp(beta * w[j, :])
                w[j, :] /= np.sum(w[j, :])
            W_prob[i] = w
        self.W_model = W_prob
        return self

    def knowledge_encoder(self, motif_name_list, motif_file, is_log_motif, autoremove_dup=False):
        from ..utils import motif
        motif_dict = motif.load_binding_motif_pssm(motif_file, is_log_motif)
        self.W_knowledge = {motif_name: motif_dict[motif_name] for motif_name in motif_name_list}
        if autoremove_dup:
            self.W_knowledge = remove_dup_motif(self.W_knowledge)
        return self

    def _build_obj_func(self):
        def obj_fn(W_model, W_knowledge, Lambda_regularizer):
            score_dict = {x: float('inf') for x in W_knowledge}
            for i in range(len(W_model)):
                w = W_model[i]
                for motif in W_knowledge:
                    d = compare_motif_diff_size(W_knowledge[motif], w)
                    if score_dict[motif] > d:
                        score_dict[motif] = d
            # K = KL + lambda * ||W||
            K = np.mean([x for x in score_dict.values()]) + Lambda_regularizer * W_model.shape[0]
            return K

        self.obj_fn = obj_fn

    def get_matched_model_weights(self):
        motif_dict = self.W_knowledge
        score_dict = {x: float('inf') for x in motif_dict}
        weight_dict = {}
        for i in range(len(self.W_model)):
            w = self.W_model[i]
            for motif in motif_dict:
                d = compare_motif_diff_size(motif_dict[motif], w)
                if score_dict[motif] > d:
                    score_dict[motif] = d
                    weight_dict[motif] = w
        return score_dict, weight_dict


class MotifSaliency(GeneralKnowledgeObjectiveFunction):
    def __init__(self,
                 output_annot,
                 session,
                 pos_prop=0.9,
                 neg_prop=0.1,
                 batch_size=None,
                 index_to_letter=None,
                 name="MotifSaliency",
                 compute_saliency_indices=True,
                 filter_motif=True,
                 normalize_to_size=None,
                 seed=777,
                 verbose=0,
                 **kwargs):
        """
        Args:
            output_annot: output_annot is a list of tuples that maps output blocks to each feature ID; the feature ID must
                also be present in the knowledge
            pos_prop:
            neg_prop:

        Attributes:
            data_record: a numpy.array for storing

        Examples:
            from BioNAS.KFunctions.MotifKnowledgeFunc import MotifSaliency, make_output_annot, get_seq_chunks
            from BioNAS.utils.plots import plot_sequence_importance
            from transfer_weights import main as make_model, read_label_annot, s
            import numpy as np

            label_annot = read_label_annot()
            cat_list = ['TF', 'Pol', 'DNase', 'Histone']
            output_annot = make_output_annot(label_annot, cat_list)

            msk = MotifSaliency(output_annot, s, batch_size=50, index_to_letter={0: 'A', 1: 'G', 2: 'C', 3: 'T'},
                                verbose=1)
            motif_file = '../../BioNAS/resources/rbp_motif/encode_motifs.txt.gz'
            msk.knowledge_encoder(motif_file=motif_file)

            model, val_data = make_model()
            print(msk(model, val_data))
        """
        self.output_annot = output_annot
        self.output_annot_w_knowledge = None
        self.pos_prop = pos_prop
        self.neg_prop = neg_prop
        self.batch_size = 32 if batch_size is None else batch_size
        self.name = name
        self.session = session
        if index_to_letter is None:
            self.index_to_letter = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        else:
            self.index_to_letter = index_to_letter
        self.saliency_ops = None
        self.is_saliency_built = False
        self.normalize_to_size = normalize_to_size or 20
        self.seed = seed
        self.data_record_x = None
        # self.data_record_seq = None
        self.data_record_map = None
        self.factor_to_labelname = None
        self.is_data_built = False
        self.verbose = verbose
        self.compute_saliency_indices = compute_saliency_indices
        self.filter_motif = filter_motif

        self._build_obj_func()

    def __call__(self, model, data, **kwargs):
        self.model_encoder(model, data)
        return self.obj_fn(self.W_model, self.W_knowledge, self)

    def _build_obj_func(self, **kwargs):
        def obj_fn(W_model, W_knowledge, msk):
            ratios = []
            for factor in msk.factor_to_labelname:
                for label_name in msk.factor_to_labelname[factor]:
                    # if self.verbose:
                    #    print(label_name)
                    ratios_ = msk.get_motif_enrichment_for_label(label_name, W_model['saliency_indices'],
                                                                 W_knowledge['motif_score'])
                    # ratios_ = msk.get_motif_dot_prod_for_label(label_name, W_model['saliency_mat'], W_knowledge['motif_score'])
                    ratios.extend(ratios_)
            kn_val = np.mean(ratios)
            # kn_val = np.median(ratios)
            return kn_val

        self.obj_fn = obj_fn
        return

    def get_motif_dot_prod_for_label(self, label_name, saliency_mat, motif_score):
        factor = label_name.split('_')[0]
        map_idx = np.stack(self.data_record_map[label_name])
        m_x = motif_score[factor][map_idx[:, 1]]
        assert len(set(map_idx[:, 0])) == 1
        block_idx = map_idx[0, 0]
        s_x = saliency_mat[block_idx][map_idx[:, 1]]
        return np.apply_along_axis(np.max, 1, s_x * m_x)

    def get_motif_enrichment_for_label(self, label_name, saliency_indices, factor_motif_score):
        factor = label_name.split('_')[0]
        ratios = []
        for map_idx in self.data_record_map[label_name]:
            pos_idx, neg_idx = saliency_indices[map_idx]
            pos_chunks = get_seq_chunks(pos_idx, min_cont=5, max_gap=3)
            neg_chunks = get_seq_chunks(neg_idx, min_cont=5, max_gap=3)
            if len(pos_chunks) == 0 or len(neg_chunks) == 0:
                ratios.append(0)
            else:
                pos_scores_max = np.max(
                    np.concatenate([factor_motif_score[factor][map_idx[1], chunk[0]:chunk[1]] for chunk in pos_chunks]))
                neg_scores_max = np.max(
                    np.concatenate([factor_motif_score[factor][map_idx[1], chunk[0]:chunk[1]] for chunk in neg_chunks]))
                # since the motif scores are already in Log Space, just use subtraction to get the fold-change
                ratio = pos_scores_max - neg_scores_max
                ratios.append(ratio)
        return ratios

    def knowledge_encoder(self, motif_file, refine_annot=True, **kwargs):
        from plots import motif
        output_factors = set([x['label_name'].split('_')[0] for x in self.output_annot])
        ref_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        swapbase = [ref_index[self.index_to_letter[i]] for i in range(4)]
        raw_motif_dict = motif.load_binding_motif_pssm(
            motif_file,
            is_log=False,
            swapbase=swapbase,
            augment_rev_comp=True
        )
        motif_dict = group_motif_by_factor(
            raw_motif_dict,
            output_factors,
            threshold=0.5,
            pure_only=self.filter_motif
        )
        self.W_knowledge = {'motif_mat': {factor_name: motif_dict[factor_name] for factor_name in motif_dict},
                            'motif_score': {}}
        output_factors_w_knowledge = output_factors.intersection(motif_dict.keys())
        self.output_annot_w_knowledge = [x for x in self.output_annot if
                                         x['label_name'].split('_')[0] in output_factors_w_knowledge]
        return self

    def model_encoder(self, model, data, **kwargs):
        if not self.is_saliency_built:
            print("building motif saliency ops..")
            saliency_ops = []
            with tf.variable_scope(self.name):
                for i in range(len(self.output_annot_w_knowledge)):
                    d = self.output_annot_w_knowledge[i]
                    y = tf.gather(model.outputs[d['block']], indices=d['index'], axis=-1)
                    saliency_op = tf.gradients(y, model.inputs)
                    # use GPU to compute/collapse base-level
                    saliency_op = tf.reduce_max(tf.abs(saliency_op[0]), axis=-1)
                    saliency_ops.append(saliency_op)
                    if i % 10 == 0: print("%i/%i" % (i, len(self.output_annot_w_knowledge)))

                # self.saliency_ops = tf.stack(saliency_ops)
                self.saliency_ops = saliency_ops
            self.is_saliency_built = True

        if not self.is_data_built:
            print("building val_data record..")
            data = list(data)
            # if type(data[0]) is not list: data[0] = [data[0]]
            if type(data[1]) is not list: data[1] = [data[1]]
            # for each output label w knowledge, extract only the positive samples
            label_pos_idx = {}
            np.random.seed(self.seed)
            for d in self.output_annot_w_knowledge:
                idx = np.where(data[1][d['block']][:, d['index']] == 1)[0]
                if len(idx):
                    # if too many, down-sample
                    if len(idx) >= self.normalize_to_size:
                        idx = np.random.choice(idx, self.normalize_to_size, replace=False)
                    # if too few, up-sample
                    else:
                        idx = np.random.choice(idx, self.normalize_to_size, replace=True)
                label_pos_idx[d['label_name']] = idx
            self.label_pos_idx = label_pos_idx
            # now collapse to get a list of all positive indices with corresponding knowledge
            total_pos_idx = sorted(list(set([i for x in label_pos_idx for i in label_pos_idx[x]])))
            self.data_record_x = data[0][total_pos_idx]
            self.data_record_y = [y[total_pos_idx] for y in data[1]]

            # keep a record of index mapping
            total_pos_idx_map = {total_pos_idx[i]: i for i in range(len(total_pos_idx))}
            # data_record_map: label_name -> (index-in-output-annot-w-knowledge, index-in-data-record)
            self.data_record_map = {}
            for i in range(len(self.output_annot_w_knowledge)):
                label_name = self.output_annot_w_knowledge[i]['label_name']
                if label_name in label_pos_idx and len(label_pos_idx[label_name]):
                    self.data_record_map[label_name] = [(i, total_pos_idx_map[x]) for x in label_pos_idx[label_name]]
            factor_to_labelname = defaultdict(list)
            for label_name in self.data_record_map:
                factor_to_labelname[label_name.split('_')[0]].append(label_name)
            self.factor_to_labelname = factor_to_labelname

            factor_motif_score = self.scan_motif()
            self.W_knowledge['motif_score'] = factor_motif_score

            self.is_data_built = True

        saliency_mat, saliency_idx = self.run_saliency(model)
        self.W_model = {
            'saliency_mat': saliency_mat,
            'saliency_indices': saliency_idx
        }
        return

    def scan_motif(self, batch_size=None, verbose=None):
        """
        :param batch_size:
        :param verbose:
        :return: dict, mapping from factors (str, e.g. CTCF) to nucleotide-level Saliency importance (np.ndarray).
            The Saliency importance array has the same dimension as self.data_record
        """
        if verbose is None:
            verbose = self.verbose
        if batch_size is None:
            batch_size = self.batch_size
        factor_to_labelname = self.factor_to_labelname
        factor_motif_score = {}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as tmp_s:
            input_seqs = tf.placeholder(shape=(None, 1000, 4), dtype=tf.float64, name='input')
            i = 0
            total = len(factor_to_labelname)
            for factor in factor_to_labelname:
                if verbose == 1:
                    print("%i/%i, %s" % (i, total, factor))
                    i += 1
                conv_out, conv_w = create_conv_from_motif(input_seqs, self.W_knowledge['motif_mat'][factor],
                                                          factor_name=factor)
                tmp_s.run(tf.variables_initializer([conv_w]))
                # self.session.run(tf.variables_initializer([conv_w]))
                tmp = []
                for x_ in batchify(self.data_record_x, batch_size=batch_size, shuffle=False, drop_remainder=False):
                    # tmp.append( self.session.run( conv_out, feed_dict={input_seqs: x_} ) )
                    tmp.append(tmp_s.run(conv_out, feed_dict={input_seqs: x_[0]}))
                tmp = np.concatenate(tmp, axis=0)
                # factor_motif_score[factor] = ((tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))).astype(np.float16)
                factor_motif_score[factor] = tmp.astype(np.float16)
        # make sure this will only clear tmp_s..
        tf.reset_default_graph()
        return factor_motif_score

    def run_saliency(self, model, batch_size=None, verbose=None):
        saliency = []
        if batch_size is None:
            batch_size = self.batch_size
        if verbose is None:
            verbose = self.verbose
        g = batchify(self.data_record_x, batch_size=batch_size, shuffle=False, drop_remainder=False)
        if verbose == 1:
            print('run saliency')
        t = trange(int(np.ceil(len(self.data_record_x) / float(batch_size)))) if verbose == 1 \
            else range(int(np.ceil(len(self.data_record_x) / float(batch_size))))
        for _ in t:
            x_ = next(g)
            tmp = self.session.run(self.saliency_ops, model._make_feed_dict(x=x_))
            tmp = np.stack(tmp)
            # with this, 1:15:28; without this, 02:56
            # tmp = [
            #    # collapse the last dim as the importance score for each locus
            #    np.apply_along_axis(np.max, 2, np.abs(tmp[i][0]))
            #    for i in range(len(self.saliency_ops))]
            saliency.append(tmp)
        saliency = np.concatenate(saliency, axis=1)

        # pre-compute the thresholds
        saliency_indices = {}
        if self.compute_saliency_indices:
            if verbose == 1:
                print('get saliency indices')
            # t2 = trange(saliency.shape[0]) if verbose else range(saliency.shape[0])
            # for i in t2:
            #    for j in range(len(saliency[i])):
            t2 = tqdm(self.data_record_map) if verbose == 1 else self.data_record_map
            for t in t2:
                for i, j in self.data_record_map[t]:
                    neg_cutoff, pos_cutoff = np.percentile(saliency[i, j, :],
                                                           [self.neg_prop * 100, self.pos_prop * 100])
                    pos_idx = np.where(saliency[i, j, :] >= pos_cutoff)[0]
                    neg_idx = np.where(saliency[i, j, :] <= neg_cutoff)[0]
                    saliency_indices[(i, j)] = [pos_idx, neg_idx]
        return saliency, saliency_indices
