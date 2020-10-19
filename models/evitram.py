import tensorflow as tf
from tensorflow.keras.layers import Input


class EvidenceTransferModel(tf.keras.Model):
    def __init__(self,
                 inputs, outputs, unet_loss, q_loss, loss_lambda, **kwargs):
        super(EvidenceTransferModel, self).__init__(inputs, outputs, **kwargs)
        self.unet_loss = unet_loss
        self.q_loss = q_loss
        self.loss_lambda = loss_lambda

    def train_step(self, data):
        x, [y, v] = data
        with tf.GradientTape() as tape:
            [y_pred, v_pred] = self(x, training=True)
            unet_loss = tf.reduce_mean(self.unet_loss(y, y_pred))
            q_loss = tf.reduce_mean(self.q_loss(v, v_pred))
            loss = unet_loss + self.loss_lambda * q_loss
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state([y, v], [y_pred, v_pred])
        return {'loss': loss, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        x, [y, v] = data
        [y_pred, v_pred] = self(x, training=False)
        unet_loss = tf.reduce_mean(self.unet_loss(y, y_pred))
        q_loss = tf.reduce_mean(self.q_loss(v, v_pred))
        loss = unet_loss + self.loss_lambda * q_loss
        self.compiled_metrics.update_state([y, v], [y_pred, v_pred])
        return {'loss': loss, **{m.name: m.result() for m in self.metrics}}


def create_evidence_transfer_model(
    unet: tf.keras.Model,
    q_model,
    connecting_layer_name
):
    connecting_layer = unet.get_layer(connecting_layer_name)

    img = Input(shape=(512, 512, 3))

    unet_intermid = tf.keras.Model(
        inputs=unet.input,
        outputs=connecting_layer.output,
        name='Unet_intermid'
    )

    x = unet(img)
    v = unet_intermid(img)
    v = q_model(v)

    return EvidenceTransferModel(
        img,
        [x, v],
        unet_loss=tf.keras.losses.categorical_crossentropy,
        q_loss=tf.keras.losses.binary_crossentropy,
        loss_lambda=1.5,
        name='EviTRAM'
    )
