from datasets.skyline12 import Skyline12
import tensorflow as tf
from models.unet import create_unet
from tensorflow.keras.layers import Input, Conv2D
from metrics import CategoricalMeanIou
import wandb
from callbacks import MyWandbCallback

skyline12 = Skyline12('datasets/skyline12/data')


def split_outputs(x, y, z):
    return (x, (y, z))


TEST_RUN = False
FOLDS = 2 if TEST_RUN else 10
train_ds = skyline12.as_tf_dataset(FOLDS, subset='train').map(split_outputs)
validation_ds = skyline12.as_tf_dataset(FOLDS, subset='validation').map(split_outputs)

import wandb
tags = ['testrun'] if TEST_RUN else []
wandb.init(project="skyline12-evidence", tags=tags, config={
    'max_epochs': 10 if TEST_RUN else 15,
    'batch_size': 64,
    'lambda': 1.5,
    'q_layers': 1,
    'q_kernel_size': 3,
    'q_activation': 'sigmoid',
    'q_loss': 'binary_crossentropy',
    'q_optimizer': 'adam',
    'q_learning_rate': 1e-5,
    'dataset': f'skyline12-folds{FOLDS}-evidence-as-img'
}, notes='continue logical-jazz-116 run for more epochs')

from models.unet import create_unet
from tensorflow.keras.layers import Input, Conv2D
from metrics import CategoricalMeanIou
from functools import partial, update_wrapper

unet = create_unet()
unet_weights = wandb.restore('model-best.h5', run_path='vassilis_krikonis/unet-baseline/3p543by5')
unet.load_weights(unet_weights.name)

img = Input(shape=(512, 512, 3), name='X')
intermid_unet = tf.keras.Model(unet.input, [unet.output, unet.layers[-2].output], name='Unet')
[x, v] = intermid_unet(img)
if wandb.config.get('q_layers') > 1:
    for i in range(wandb.config.get('q_layers') - 1):
        v = Conv2D(64, (3,3), padding='same', activation='relu', name=f'Q_layer_{i}')(v)
v = Conv2D(
    1,
    (wandb.config.get('q_kernel_size'), wandb.config.get('q_kernel_size')),
    padding='same',
    kernel_initializer=wandb.config.get('q_kernel_initializer'),
    activation=wandb.config.get('q_activation'),
    name='Q'
)(v)
evid_model = tf.keras.Model(inputs=img, outputs=[x, v])
if wandb.config.get('q_optimizer') == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.get('q_learning_rate'))
elif wandb.config.get('q_optimizer') == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.get('q_learning_rate'), momentum=0.99)
else:
    optimizer = wandb.config.get('q_optimizer')

q_loss = wandb.config.get('q_loss')
if q_loss == 'psnr':
    def psnr_loss_fn(y_preds, y):
        y_preds = tf.cast(y_preds, tf.float32)
        y = tf.cast(y, tf.float32)
        return tf.image.psnr(y_preds, y, max_val=1.0)
    q_loss = psnr_loss_fn
evid_model.compile(
    optimizer=optimizer,
    loss=['categorical_crossentropy', q_loss],
    loss_weights=[1.0, wandb.config.get('lambda')],
    metrics=[[CategoricalMeanIou(num_classes=5), 'accuracy'], ['accuracy']],
    run_eagerly=False
)

import wandb
from callbacks import MyWandbCallback

early_stopper_Unet = tf.keras.callbacks.EarlyStopping(
    monitor='val_Unet_categorical_mean_iou',
    mode='min',
    patience=5, 
    restore_best_weights=True
)
early_stopper_Q = tf.keras.callbacks.EarlyStopping(
    monitor='val_Q_loss',
    mode='min',
    patience=5, 
    restore_best_weights=True
)
data_to_log = next(iter(validation_ds.batch(10)))

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = wandb.config.get('batch_size')
evid_model.fit(
    train_ds.batch(BATCH_SIZE).cache(f'temp/train_{FOLDS}_xyz_img').prefetch(AUTOTUNE),
    epochs=wandb.config.get('max_epochs'),
    validation_data=validation_ds.batch(BATCH_SIZE).cache(f'temp/val_{FOLDS}_xyz_img').prefetch(AUTOTUNE),
    callbacks=[
        early_stopper_Unet,
        early_stopper_Q,
        MyWandbCallback(
            include='xz' if TEST_RUN else 'xyz',
            val_data=data_to_log,
            save_model=True,
            save_weights_only=True,
            input_type='image',
            output_type='segmentation_mask'),
    ],
    verbose=2
)


