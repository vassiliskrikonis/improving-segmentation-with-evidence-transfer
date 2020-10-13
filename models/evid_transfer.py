import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Reshape, Flatten, Input, LeakyReLU, Conv2DTranspose
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
import ipdb


class EvidenceTransferModel(tf.keras.Model):
    def train_step(self, data):
        x, [y, z] = data
        with tf.GradientTape() as unet_tape, tf.GradientTape() as q_tape:
            [y_pred, z_pred] = self(x, training=True)
            pr_loss = tf.reduce_mean(categorical_crossentropy(y, y_pred))
            q_loss = tf.reduce_mean(binary_crossentropy(z, z_pred))
            loss = pr_loss + 1.5 * q_loss

        # update unet weights w.r.t combined loss
        unet_vars = [
            *self.get_layer('Unet_intermid').trainable_variables,
            *self.get_layer('Unet_last').trainable_variables
        ]
        unet_grads = unet_tape.gradient(loss, unet_vars)
        self.optimizer.apply_gradients(zip(unet_grads, unet_vars))

        # update Q weights w.r.t combined loss
        q_vars = self.get_layer('Q').trainable_variables
        q_grads = q_tape.gradient(loss, q_vars)
        self.optimizer.apply_gradients(zip(q_grads, q_vars))

        self.compiled_metrics.update_state([y, z], [y_pred, z_pred])
        return {'loss': loss, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        x, [y, z] = data
        [y_pred, z_pred] = self(x, training=False)
        pr_loss = tf.reduce_mean(categorical_crossentropy(y, y_pred))
        q_loss = tf.reduce_mean(binary_crossentropy(z, z_pred))
        loss = pr_loss + 1.5 * q_loss
        self.compiled_metrics.update_state([y, z], [y_pred, z_pred])
        return {'loss': loss, **{m.name: m.result() for m in self.metrics}}


def create_q_model(config, input_shape=(512, 512, 64)):
    v_in = Input(shape=input_shape)
#     v = Conv2D(input_shape[-1], (3, 3), strides=(1, 1), padding='same', activation='relu')(v_in)
    v = Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='relu')(v_in)
    model = tf.keras.Model(inputs=v_in, outputs=v, name='Q')
    print(model.summary())
    return model


# def create_q_model(config, input_shape=(512, 512, 64)):
#     img = Input(shape=input_shape)
#     x = img
#     x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2DTranspose(1, (3, 3), strides=2, padding='same', activation='sigmoid')(x)
#     model = tf.keras.Model(inputs=img, outputs=x, name='Q')
#     print(model.summary())
#     return model


def create_evidence_transfer_model(unet, q_model, config):
    unet_intermid = tf.keras.Model(inputs=unet.input, outputs=unet.layers[-2].output, name='Unet_intermid')
    unet_last_layer = unet.layers[-1]

    feat_map = Input(shape=(512, 512, 64))
    x = unet_last_layer(feat_map)
    unet_last = tf.keras.Model(inputs=feat_map, outputs=x, name='Unet_last')

    img = Input(shape=(512, 512, 3))
    x = unet_intermid(img)
    v = q_model(x)
    x = unet_last(x)

    return EvidenceTransferModel(inputs=img, outputs=[x, v], name='EvidTran')
