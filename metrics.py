import tensorflow as tf


class CategoricalMeanIou(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)