import tensorflow as tf


class LDAMLoss(tf.keras.losses.Loss):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, reduction = tf.keras.loses.Reduction.AUTO, name="LDAM"):
        super().__init__(reduction=reduction, name=name)
        m_list = 1.0 /np.sqrt(np.sqrt(cls_num_list))
        m_list *=  max_m(m_list.max())
        self.m_list = tf.convert_to_tensor(m_list, tf.float32)
        assert s > 0
        self.s=s
        self.w=weight

    def __call__(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
