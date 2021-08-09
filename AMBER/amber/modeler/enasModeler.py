# -*- coding: UTF-8 -*-

import warnings
from .dag import get_dag, get_layer
from .dag import ComputationNode
from .dag import EnasConv1dDAG

class ModelBuilder:
    """Scaffold of Model Builder
    """

    def __init__(self, inputs, outputs, *args, **kwargs):
        raise NotImplementedError("Abstract method.")

    def __call__(self, model_states):
        raise NotImplementedError("Abstract method.")


class DAGModelBuilder(ModelBuilder):
    def __init__(self, inputs_op, output_op,
                 model_space, model_compile_dict,
                 num_layers=None,
                 with_skip_connection=True,
                 with_input_blocks=True,
                 dag_func=None,
                 *args, **kwargs):

        if type(inputs_op) not in (list, tuple):
            self.inputs_op = [inputs_op]
            warnings.warn("inputs_op should be list-like; if only one input, try using ``[inputs_op]`` as argument",
                         stacklevel=2)
        else:
            self.inputs_op = inputs_op
        self.output_op = output_op
        self.model_space = model_space
        self.num_layers = num_layers or len(self.model_space)
        self.model_compile_dict = model_compile_dict
        self.with_input_blocks = with_input_blocks
        self.with_skip_connection = with_skip_connection
        self.dag_func_ = dag_func
        self.dag_func = get_dag(dag_func) if dag_func is not None else DAG

    def __str__(self):
        s = 'DAGModelBuilder with builder %s' % self.dag_func_
        return s

    def __call__(self, arc_seq, *args, **kwargs):
        input_nodes = self._get_input_nodes()
        output_node = self._get_output_node()
        dag = self.dag_func(arc_seq=arc_seq,
                            num_layers=self.num_layers,
                            model_space=self.model_space,
                            input_node=input_nodes,
                            output_node=output_node,
                            with_skip_connection=self.with_skip_connection,
                            with_input_blocks=self.with_input_blocks,
                            *args,
                            **kwargs)
        try:
            model = dag._build_dag()
            model.compile(**self.model_compile_dict)
        except ValueError:
            print(arc_seq)
            raise Exception('above')
        return model

    def _get_input_nodes(self):
        input_nodes = []
        for node_op in self.inputs_op:
            node = ComputationNode(node_op, node_name=node_op.Layer_attributes['name'])
            input_nodes.append(node)
        return input_nodes

    def _get_output_node(self):
        if type(self.output_op) is list:
            raise Exception("DAG currently does not accept output_op in List")
        output_node = ComputationNode(self.output_op, node_name='output')
        return output_node


class EnasAnnModelBuilder(DAGModelBuilder):
    """This function builds an Artificial Neural net.

    It uses tensorflow low-level API to define a big graph, where
    each child network architecture is a subgraph in this big DAG.

    Parameters
    ----------
    session
    controller
    dag_func
    l1_reg
    l2_reg
    with_output_blocks
    use_node_dag
    feature_model
    dag_kwargs
    args
    kwargs
    """
    def __init__(self, session=None, controller=None, dag_func='EnasAnnDAG', l1_reg=0.0, l2_reg=0.0,
                 with_output_blocks=False,
                 use_node_dag=True,
                 feature_model=None,
                 dag_kwargs=None,
                 *args,
                 **kwargs):
        super().__init__(dag_func=dag_func, *args, **kwargs)
        assert dag_func.lower() in ('enas', 'enasanndag'), "EnasModelBuilder only support enasDAG."
        self.session = session
        self.controller = controller
        self.l1_reg = float(l1_reg)
        self.l2_reg = float(l2_reg)
        # -----
        # BELOW ARE NEW ARGS; Enas-specific args
        self.node_dag = None
        self.use_node_dag = use_node_dag
        self.feature_model = feature_model
        self.dag_kwargs = dag_kwargs or {}
        self.with_output_blocks = with_output_blocks
        assert not (
                    self.with_output_blocks and self.use_node_dag), "Currently `use_node_dag` is incompatible with `with_output_blocks`"
        # END NEW ARGS
        # -----
        self._build_dag()

    def _build_dag(self):
        """
        Args:
            *args:
            **kwargs: When needed, `feature_model` will be parsed through kwargs to DAG

        Returns:

        """
        self.dag = self.dag_func(model_space=self.model_space,
                                 input_node=self.inputs_op,
                                 output_node=self.output_op,
                                 with_input_blocks=self.with_input_blocks,
                                 with_skip_connection=self.with_skip_connection,
                                 with_output_blocks=self.with_output_blocks,
                                 session=self.session,
                                 model_compile_dict=self.model_compile_dict,
                                 l1_reg=self.l1_reg,
                                 l2_reg=self.l2_reg,
                                 controller=self.controller,
                                 feature_model=self.feature_model,
                                 **self.dag_kwargs)

    def __call__(self, arc_seq=None, *args, **kwargs):
        input_nodes = self._get_input_nodes()
        output_node = self._get_output_node()
        if self.use_node_dag:
            self.node_dag = lambda x: get_dag('InputBlockDAG')(
                arc_seq=x,
                num_layers=self.num_layers,
                model_space=self.model_space,
                input_node=input_nodes,
                output_node=output_node,
                with_skip_connection=self.with_skip_connection,
                with_input_blocks=self.with_input_blocks,
            )
        model = self.dag(arc_seq, node_builder=self.node_dag)
        model.compile(**self.model_compile_dict)
        return model

    def set_controller(self, controller):
        self.dag.set_controller(controller)

    # overwrite
    def _get_output_node(self):
        if type(self.output_op) is list:
            output_node = [ComputationNode(self.output_op[i], node_name=self.output_op[i].Layer_attributes['name'])
                           for i in range(len(self.output_op))]
        else:
            output_node = [ComputationNode(self.output_op, node_name='output')]
        return output_node


class EnasCnnModelBuilder(DAGModelBuilder):
    def __init__(self, session=None, controller=None, dag_func='EnasConv1DDAG', l1_reg=0.0, l2_reg=0.0,
                 batch_size=None,
                 dag_kwargs=None,
                 *args,
                 **kwargs):
        """
        Args:
            session:
            controller:
            dag_func:
            l1_reg:
            use_node_dag: if True, will use InputBlockDAG to build a list of `ComputationNode`s to record the parent/
                child relationships; otherwise, do not use node_dag.
                Currently `use_node_dag=True` is incompatible with `with_output_blocks=True`
            *args:
            **kwargs:
        """
        super().__init__(dag_func=dag_func, *args, **kwargs)
        #assert dag_func.lower() in ('enascnndag', 'enasconv1ddag'), "EnasModelBuilder only support enasDAG."
        self.session = session
        self.controller = controller
        self.l1_reg = float(l1_reg)
        self.l2_reg = float(l2_reg)
        # -----
        # BELOW ARE NEW ARGS; Enas-specific args
        self.batch_size = batch_size or 128
        self.dag_kwargs = dag_kwargs or {}
        # END NEW ARGS
        # -----
        self._build_dag()
        assert issubclass(type(self.dag), EnasConv1dDAG), "EnasModelBuilder only support enasDAG and its derivatives"

    def _build_dag(self):
        """
        Args:
            *args:
            **kwargs: When needed, `feature_model` will be parsed through kwargs to DAG

        Returns:

        """
        self.dag = self.dag_func(model_space=self.model_space,
                                 input_node=self.inputs_op,
                                 output_node=self.output_op,
                                 session=self.session,
                                 model_compile_dict=self.model_compile_dict,
                                 l1_reg=self.l1_reg,
                                 l2_reg=self.l2_reg,
                                 controller=self.controller,
                                 batch_size=self.batch_size,
                                 **self.dag_kwargs
                                 )

    def __call__(self, arc_seq=None, *args, **kwargs):
        model = self.dag(arc_seq, **kwargs)
        model.compile(**self.model_compile_dict)
        return model

    def set_controller(self, controller):
        self.dag.set_controller(controller)
