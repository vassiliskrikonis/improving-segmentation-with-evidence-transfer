from datasets.skyline12 import Skyline12
import tensorflow as tf
from models.unet import create_unet
from tensorflow.keras.layers import Input, Conv2D
from metrics import CategoricalMeanIou
import wandb
from callbacks import MyWandbCallback

skyline12 = Skyline12('/storage/skyline12/data')


def split_outputs(x, y, z):
    return (x, (y, z))


FOLDS = 2
train_ds = skyline12.as_tf_dataset(
    FOLDS, subset='train').map(split_outputs)
validation_ds = skyline12.as_tf_dataset(
    FOLDS, subset='validation').map(split_outputs)


wandb.init(project="skyline12-evidence", tags=[], config={
    'max_epochs': 50,
    'lambda': 1.5,
    'q_activation': 'tanh',
    'q_loss': 'binary_crossentropy',
    'q_optimizer': 'adam',
    'q_learning_rate': 0.00001,
    'dataset': f'skyline12-folds{FOLDS}-evidence-as-img'
})

unet = create_unet()
unet_weights = wandb.restore('model-best.h5', run_path='vassilis_krikonis/unet-baseline/3p543by5')
unet.load_weights(unet_weights.name)

img = Input(shape=(512, 512, 3), name='X')
intermid_unet = tf.keras.Model(unet.input, [unet.output, unet.layers[-2].output], name='Unet')
[x, v] = intermid_unet(img)
v = Conv2D(
    2,
    (1, 1),
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
evid_model.compile(
    optimizer=optimizer,
    loss=['categorical_crossentropy', wandb.config.get('q_loss')],
    loss_weights=[1.0, wandb.config.get('lambda')],
    metrics=[[CategoricalMeanIou(num_classes=5), 'accuracy'], ['accuracy']],
    run_eagerly=False
)

early_stopper = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
data_to_log = next(iter(validation_ds.batch(10)))

AUTOTUNE = tf.data.experimental.AUTOTUNE
evid_model.fit(
    train_ds.batch(3).cache(f'temp/train_{FOLDS}_xyz_img').prefetch(AUTOTUNE),
    epochs=wandb.config.get('max_epochs'),
    validation_data=validation_ds.batch(3).cache(f'temp/val_{FOLDS}_xyz_img').prefetch(AUTOTUNE),
    callbacks=[
        early_stopper,
        MyWandbCallback(
            val_data=data_to_log,
            include='xyz',
            save_model=True,
            save_weights_only=True,
            input_type='image',
            output_type='segmentation_mask')
    ]
)


