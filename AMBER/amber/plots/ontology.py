# -*- encoding: utf-8 -*-

"""
utils for ontology NAS
"""

import matplotlib.pyplot as plt
import networkx as nx
# from GenoPheno2 import SubSystem
from networkx.drawing.nx_agraph import graphviz_layout

from .plotsV1 import reset_plot


def plot_nx_dag(ss, save_fn=None):
    reset_plot()
    g = nx.DiGraph()
    g.add_edges_from(
        [
            (i, ss.nodes[i].child[j].id)
            for i in ss.nodes
            for j in range(len(ss.nodes[i].child))
        ]
    )
    pos = graphviz_layout(g, prog='dot')
    nx.draw_networkx_nodes(g, pos, cmap=plt.get_cmap('jet'), node_size=500)
    nx.draw_networkx_labels(g, pos, font_size=7.5)
    nx.draw_networkx_edges(g, pos, arrow=True)
    if save_fn:
        plt.savefig(save_fn)
    else:
        plt.show()
