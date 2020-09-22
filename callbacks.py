import tensorflow as tf
import wandb


class WandbLogPredictions(tf.keras.callbacks.Callback):
    def __init__(self, val_data, class_labels):
        self.val_data = val_data
        self.class_labels = class_labels

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.val_data
        preds = self.model(x, training=False)
        wandb_images = [wandb.Image(img, masks={
            'predictions': {
                'mask_data': pred.numpy().argmax(-1),
                'class_labels': self.class_labels
            },
        }) for img, pred in zip(x, preds)]
        wandb.log({'predictions': wandb_images}, epoch)