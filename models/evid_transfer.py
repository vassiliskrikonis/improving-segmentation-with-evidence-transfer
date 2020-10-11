import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy


class EvidenceTransferModel(tf.keras.Model):
    def train_step(self, data):
        x, [y, z] = data

        with tf.GradientTape(persistent=True) as tape:
            [y_pred, z_pred] = self(x, training=True)
            pr_loss = categorical_crossentropy(y, y_pred)
            q_loss = binary_crossentropy(z, z_pred)
            loss = pr_loss + 1.0 * q_loss

        # unet_vars = self.get_layer('Unet').trainable_variables
        # q_vars = self.get_layer('Q').trainable_variables
        grads = tape.gradient(loss, self.trainable_variables)
        grads = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state([y, z], [y_pred, z_pred])
        return {**{'loss': loss}, **{m.name: m.result() for m in self.metrics}}


def create_q_model(config):
    v_in = Input(shape=(512, 512, 64))
    v = v_in
    if config.get('q_layers') > 1:
        for i in range(config.get('q_layers') - 1):
            v = Conv2D(64, (3, 3), padding='same',
                       activation='relu', name=f'Q_layer_{i}')(v)
    v = Conv2D(
        1,
        (config.get('q_kernel_size'), config.get('q_kernel_size')),
        padding='same',
        kernel_initializer=config.get('q_kernel_initializer'),
        activation=config.get('q_activation'),
        name='Q'
    )(v)
    return tf.keras.Model(inputs=v_in, outputs=v, name='Q')


def create_evidence_transfer_model(pr_model, q_model, config):
    intermid_pr_model = tf.keras.Model(
        pr_model.input,
        [pr_model.output, pr_model.layers[-2].output],
        name='Unet')

    img = Input(shape=(512, 512, 3), name='X')
    [x, v] = intermid_pr_model(img)
    v = q_model(v)

    return EvidenceTransferModel(inputs=img, outputs=[x, v], name='Evidence Transfer')
