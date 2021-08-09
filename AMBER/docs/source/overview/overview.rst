Overview of AMBER
==================

.. _figure_1:
.. figure:: /_static/img/controller.png
    :align: center
    :alt: controller RNN
    :figclass: align-center

    **Figure 1**

AMBER is a toolkit for designing neral network models automatically (modified from Zoph et al., 2017).

Why AMBER?
------------
Convolutional Neural Networks (CNNs) have become a standard for analysis of biological sequences.
Tuning of network architectures is essential for CNN’s performance,
yet it requires substantial knowledge of machine learning and commitment of time and effort.
This process thus imposes a major barrier to broad and effective application of modern deep learning in genomics.

AMBER is a fully automated framework to efficiently design and apply CNNs for genomic sequences, which stands for
Automated Modeling for Biological Evidence-based Research. It designs optimal models for user-specified biological
questions through the state-of-the-art Neural Architecture Search (NAS). With AMBER, it provides an efficient
automated method for designing accurate deep learning models in genomics.


The essence of AMBER
--------------------

As shown in :ref:`figure_1`, there are essentially two components in AMBER (and many other NAS methods):

    - An ``Architect`` that designs model architectures from its past experiences;
    - A ``Modeler`` that implements the architect's architectures, and collects feedbacks.

In AMBER, these two components are implemented as two submodules. Before we go into tedious technical details, the high-
level idea is quite simple:

    "Architects designs architectures, and modelers implement architecture into models to collect each model's performance.
    Through the feedback loops between architects and modelers, we hope that, over time, the designated architect can
    design better/more educated model architectures by learning from the history/past experience."

With that in mind, now we start to look into what architects and modelers are currently available in AMBER, and what helpers
they need to work together.

How to Cite
-------------
If you find AMBER useful in your work, please cite our paper:

1. Z Zhang, CY Park, CL Theesfeld, OG Troyanskaya (2021).
"An automated framework for efficiently designing deep convolutional neural networks in genomics". Nature Machine Intelligence. Mar 15:1-9.
[`Cover for Nature Machine Intelligence, Vol. 3, Issue 5 <https://www.nature.com/natmachintell/volumes/3/issues/5>`_]
[`News&Views by Zhang et al. <https://www.nature.com/articles/s42256-021-00350-x>`_]
