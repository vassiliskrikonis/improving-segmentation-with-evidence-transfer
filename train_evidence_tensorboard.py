#%%
from models.unet import create_unet
import tensorflow as tf
from datasets.skyline12 import Skyline12
from metrics import CategoricalMeanIou
from models.evid_transfer import create_evidence_transfer_model, create_q_model
from tensorflow.keras.callbacks import TensorBoard
import datetime

AUTOTUNE = tf.data.experimental.AUTOTUNE

#%%
skyline12 = Skyline12('datasets/skyline12/data')


def split_outputs(x, y, z):
    return (x, (y, z))


TEST_RUN = True
FOLDS = 2 if TEST_RUN else 10
train_ds = skyline12.as_tf_dataset(FOLDS, subset='train').map(
    split_outputs, num_parallel_calls=AUTOTUNE)
validation_ds = skyline12.as_tf_dataset(FOLDS, subset='validation').map(
    split_outputs, num_parallel_calls=AUTOTUNE)

#%%
hparams = {
    'max_epochs': 5 if TEST_RUN else 15,
    'batch_size': 3,
    'lambda': 1.5,
    'q_layers': 1,
    'q_kernel_size': 3,
    'q_activation': 'sigmoid',
    'q_loss': 'binary_crossentropy',
    'q_optimizer': 'adam',
    'q_learning_rate': 1e-5,
    'dataset': f'skyline12-folds{FOLDS}-evidence-as-img'
}

#%%
unet = create_unet()
unet.load_weights('model-best.h5')
q_model = create_q_model(hparams)
evid_model = create_evidence_transfer_model(unet, q_model, hparams)

#%%
if hparams.get('q_optimizer') == 'adam':
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams.get('q_learning_rate'))
elif hparams.get('q_optimizer') == 'sgd':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=hparams.get('q_learning_rate'), momentum=0.99)
else:
    optimizer = hparams.get('q_optimizer')
q_loss = hparams.get('q_loss')
if q_loss == 'psnr':
    def psnr_loss_fn(y_preds, y):
        y_preds = tf.cast(y_preds, tf.float32)
        y = tf.cast(y, tf.float32)
        return tf.image.psnr(y_preds, y, max_val=1.0)
    q_loss = psnr_loss_fn

evid_model.compile(
    optimizer=optimizer,
    metrics=[[CategoricalMeanIou(num_classes=5), 'accuracy'], ['accuracy']],
    run_eagerly=True
)

#%%
log_dir = 'temp/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir)

#%%
BATCH_SIZE = hparams.get('batch_size')
evid_model.fit(
    train_ds.batch(BATCH_SIZE).cache(
        f'temp/train_{FOLDS}_xyz_img').prefetch(AUTOTUNE),
    epochs=hparams.get('max_epochs'),
    validation_data=validation_ds.batch(BATCH_SIZE).cache(
        f'temp/val_{FOLDS}_xyz_img').prefetch(AUTOTUNE),
    callbacks=[
        tensorboard_cb
    ],
    verbose=2
)

#%%
import matplotlib.pyplot as plt
from itertools import islice
for x, [y,z] in validation_ds.batch(1).take(5):
    [y_preds, z_preds] = evid_model(x, training=False)
    Skyline12.show_sample(x[0], [y_preds[0], z_preds[0]], from_tensors=True)

# %%
