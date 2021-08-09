"""
Test enas model builder
"""

import tensorflow as tf
import numpy as np

from amber.utils import testing_utils
from amber import modeler
from amber import architect


class TestEnasConvModeler(testing_utils.TestCase):
    def setUp(self):
        self.session = tf.Session()
        self.input_op = [architect.Operation('input', shape=(10, 4), name="input")]
        self.output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
        self.x = np.random.choice(2, 40).reshape((1, 10, 4))
        self.y = np.random.sample(1).reshape((1, 1))
        self.model_space, _ = testing_utils.get_example_conv1d_space()
        self.model_compile_dict = {'loss': 'binary_crossentropy', 'optimizer': 'sgd'}
        self.controller = architect.GeneralController(
            model_space=self.model_space,
            buffer_type='ordinal',
            with_skip_connection=True,
            kl_threshold=0.05,
            buffer_size=15,
            batch_size=5,
            session=self.session,
            train_pi_iter=2,
            lstm_size=32,
            lstm_num_layers=1,
            lstm_keep_prob=1.0,
            optim_algo="adam",
            skip_target=0.8,
            skip_weight=0.4,
        )
        self.target_arc = [0, 0, 1]
        self.enas_modeler = modeler.EnasCnnModelBuilder(
            model_space=self.model_space,
            num_layers=len(self.model_space),
            inputs_op=self.input_op,
            output_op=self.output_op,
            model_compile_dict=self.model_compile_dict,
            session=self.session,
            controller=self.controller,
            batch_size=1,
            dag_kwargs={
                'stem_config': {
                    'has_stem_conv': True,
                    'fc_units': 5
                }
            }
        )
        self.num_samps = 15

    def test_sample_arc_builder(self):
        model = self.enas_modeler()
        samp_preds = [model.predict(self.x).flatten()[0] for _ in range(self.num_samps)]
        # sampled loss can't be always identical
        self.assertNotEqual(len(set(samp_preds)), 1)
        old_loss = [model.evaluate(self.x, self.y)['val_loss'] for _ in range(self.num_samps)]
        model.fit(self.x, self.y, batch_size=1, epochs=100, verbose=0)
        new_loss = [model.evaluate(self.x, self.y)['val_loss'] for _ in range(self.num_samps)]
        self.assertLess(sum(new_loss), sum(old_loss))

    def test_fix_arc_builder(self):
        model = self.enas_modeler(arc_seq=self.target_arc)
        # fixed preds must always be identical
        fix_preds = [model.predict(self.x).flatten()[0] for _ in range(self.num_samps)]
        self.assertEqual(len(set(fix_preds)), 1)
        # record original loss
        old_loss = model.evaluate(self.x, self.y)['val_loss']
        # train weights with sampled arcs from model2
        model2 = self.enas_modeler()
        model2.fit(self.x, self.y, batch_size=1, epochs=100, verbose=0)
        # loss should reduce
        new_loss = model.evaluate(self.x, self.y)['val_loss']
        self.assertLess(new_loss, old_loss)
        # fixed preds should still be identical
        fix_preds = [model.predict(self.x).flatten()[0] for _ in range(self.num_samps)]
        self.assertEqual(len(set(fix_preds)), 1)


if __name__ == '__main__':
    tf.test.main()
