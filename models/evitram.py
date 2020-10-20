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
        with tf.GradientTape() as unet_tape, tf.GradientTape() as q_tape:
            [y_pred, v_pred] = self(x, training=True)
            unet_loss = tf.reduce_mean(self.unet_loss(y, y_pred))
            q_loss = tf.reduce_mean(self.q_loss(v, v_pred))
            loss = unet_loss + self.loss_lambda * q_loss

        # apply gradients of loss wrt unet weights
        unet_vars = self.get_layer('Unet').trainable_variables
        unet_grads = unet_tape.gradient(loss, unet_vars)
        self.optimizer.apply_gradients(zip(unet_grads, unet_vars))

        # apply gradients of loss wrt q weights
        q_vars = self.get_layer('Q').trainable_variables
        q_grads = q_tape.gradient(loss, q_vars)
        self.optimizer.apply_gradients(zip(q_grads, q_vars))

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
    q = tf.keras.Sequential([
        unet_intermid,
        q_model
    ], name='Q')

    x = unet(img)
    v = q(img)

    return EvidenceTransferModel(
        img,
        [x, v],
        unet_loss=tf.keras.losses.categorical_crossentropy,
        q_loss=tf.keras.losses.binary_crossentropy,
        loss_lambda=1.5,
        name='EviTRAM'
    )
