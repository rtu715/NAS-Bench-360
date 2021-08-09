# -*- coding: UTF-8 -*-

from __future__ import print_function

import gzip
from subprocess import *

import numpy as np

try:
    import matplotlib as mpl

    mpl.use('Agg')
    import seaborn
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-ticks')
    from matplotlib import transforms
    import matplotlib.patheffects
    from matplotlib.font_manager import FontProperties

    do_plot = True
except:
    do_plot = False


def read_file(filename):
    if filename.endswith('gz'):
        with gzip.GzipFile(filename) as fp:
            List = [x.strip().decode('utf-8') for x in fp if len(x.strip()) > 0]
    else:
        with open(filename) as fp:
            List = [x.strip() for x in fp if len(x.strip()) > 0]
    return List


def load_binding_motif_pssm(motif_file, is_log, swapbase=None, augment_rev_comp=False):
    dict_binding_pssm = {}
    rbp, pssm = '', []
    for line in read_file(motif_file):
        if line.startswith('#'):
            continue
        if line.startswith('>'):
            if len(pssm) > 0:
                if is_log:
                    pssm = convertlog2freq(pssm)
                pssm = np.array(pssm)
                if augment_rev_comp:
                    # print("augment rev comp")
                    pssm_rc = pssm[::-1, ::-1]
                if swapbase:
                    # print("swapbase: %s"%swapbase)
                    newbase = swapbase
                else:
                    newbase = [0, 1, 2, 3]
                dict_binding_pssm[rbp] = pssm[:, newbase]
                if augment_rev_comp:
                    dict_binding_pssm[rbp + '_rc'] = pssm_rc[:, newbase]
            rbp = line.strip()[1:].split()[0]
            pssm = []
        else:
            # pssm.append([float(x) for x in line.strip().split('\t')] + [0])
            ele = line.strip().split()
            column_prob = [float(x) for x in ele[-4:]]
            pssm.append(column_prob)
    return dict_binding_pssm


def convertlog2freq(pssm):
    pssm = np.array(pssm)
    pssm = np.power(2, pssm) * 0.25
    pssm = pssm[:, 0: 4].T
    # pssm = pssm - np.amin(pssm, axis = 0)
    pssm = pssm / np.sum(pssm, axis=0)
    return pssm.T


def draw_dnalogo_Rscript(pssm, savefn='seq_logo.pdf'):
    width = 4. / 8. * pssm.shape[1]
    pssm_flatten = pssm.flatten()
    seq_len = len(pssm[0])
    fw = open('/tmp/draw.seq.log.R', 'w')
    fw.write('seq_profile = c(' + ','.join([str(x) for x in pssm_flatten]) + ')' + '\n')
    fw.write('seq_matrix = matrix(seq_profile, 4, {}, byrow = T)'.format(seq_len) + '\n')
    fw.write("rownames(seq_matrix) <- c('A', 'C', 'G', 'T')" + '\n')
    fw.write("library(ggseqlogo)" + '\n')
    fw.write("library(ggplot2)" + '\n')
    fw.write("p <- ggplot() + geom_logo(seq_matrix) + theme_logo() + theme(axis.text.x = element_blank(), " + '\n')
    fw.write(
        "                                                                                                       panel.spacing = unit(0.5, 'lines')," + '\n')
    fw.write(
        "                                                                                                       axis.text.y = element_blank(), " + '\n')
    fw.write(
        "                                                                                                       axis.title.y = element_blank()," + '\n')
    fw.write(
        "                                                                                                       axis.title.x = element_blank(), " + '\n')
    fw.write(
        "                                                                                                       plot.title = element_text(hjust = 0.5, size = 20)," + '\n')
    fw.write(
        "                                                                                                       legend.position = 'none') + ggtitle('seqlogo')" + '\n')
    fw.write("ggsave('%s', units='in', width=%i, height=4 )\n" % (savefn, width))
    fw.close()
    cmd = 'Rscript /tmp/draw.seq.log.R'
    call(cmd, shell=True)


def draw_dnalogo_matplot(pssm):
    all_scores = []
    letter_idx = ['A', 'C', 'G', 'T']
    for row in pssm:
        tmp = []
        for i in range(4):
            tmp.append((letter_idx[i], row[i]))
        all_scores.append(sorted(tmp, key=lambda x: x[1]))
    draw_logo(all_scores)


class Scale(matplotlib.patheffects.RendererBase):
    '''http://nbviewer.jupyter.org/github/saketkc/notebooks/blob/master/python/Sequence%20Logo%20Python%20%20--%20Any%20font.ipynb?flush=true
    ## Author: Saket Choudhar [saketkc\\gmail]
    ## License: GPL v3
    ## Copyright © 2017 Saket Choudhary<saketkc__AT__gmail>
    '''

    def __init__(self, sx, sy=None):
        self._sx = sx
        self._sy = sy

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        affine = affine.identity().scale(self._sx, self._sy) + affine
        renderer.draw_path(gc, tpath, affine, rgbFace)


def draw_logo(all_scores, fontfamily='Arial', size=80,
              COLOR_SCHEME={'G': 'orange', 'A': 'red', 'C': 'blue', 'T': 'darkgreen'}):
    """
    References
    ----------
    http://nbviewer.jupyter.org/github/saketkc/notebooks/blob/master/python/Sequence%20Logo%20Python%20%20--%20Any%20font.ipynb?flush=true

    Author: Saket Choudhar [saketkc\\gmail]

    License: GPL v3

    Copyright © 2017 Saket Choudhary<saketkc__AT__gmail>
    """
    if fontfamily == 'xkcd':
        plt.xkcd()
    else:
        mpl.rcParams['font.family'] = fontfamily

    fig, ax = plt.subplots(figsize=(len(all_scores), 2.5))

    font = FontProperties()
    font.set_size(size)
    font.set_weight('bold')

    # font.set_family(fontfamily)

    ax.set_xticks(range(1, len(all_scores) + 1))
    ax.set_yticks(range(0, 3))
    ax.set_xticklabels(range(1, len(all_scores) + 1), rotation=90)
    ax.set_yticklabels(np.arange(0, 3, 1))
    seaborn.despine(ax=ax, trim=True)

    trans_offset = transforms.offset_copy(ax.transData,
                                          fig=fig,
                                          x=1,
                                          y=0,
                                          units='dots')

    for index, scores in enumerate(all_scores):
        yshift = 0
        for base, score in scores:
            txt = ax.text(index + 1,
                          0,
                          base,
                          transform=trans_offset,
                          fontsize=80,
                          color=COLOR_SCHEME[base],
                          ha='center',
                          fontproperties=font,

                          )
            txt.set_path_effects([Scale(1.0, score)])
            fig.canvas.draw()
            window_ext = txt.get_window_extent(txt._renderer)
            yshift = window_ext.height * score
            trans_offset = transforms.offset_copy(txt._transform,
                                                  fig=fig,
                                                  y=yshift,
                                                  units='points')
        trans_offset = transforms.offset_copy(ax.transData,
                                              fig=fig,
                                              x=1,
                                              y=0,
                                              units='points')
    plt.show()
