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
        # commit == False because WandbCallback doesn't use epochs as steps
        # important! just make sure to use this callback before WandbCallback
        wandb.log({'predictions': wandb_images}, commit=False)


class WandbLogEvidencePredictions(tf.keras.callbacks.Callback):
    def __init__(self, val_data, class_labels):
        self.val_data = val_data
        self.class_labels = class_labels

    def on_epoch_end(self, epoch, logs=None):
        x, [y, v] = self.val_data
        preds = self.model(x, training=False)
        wandb_images = [wandb.Image(img, masks={
            'segmap': {
                'mask_data': pred_y.numpy().argmax(-1),
                'class_labels': self.class_labels
            },
            'evidence': {
                'mask_data': pred_v.numpy().argmax(-1),
                'class_labels': {0: 'nil', 1: 'scribble'}
            }
        }) for img, pred_y, pred_v in zip(x, *preds)]
        # commit == False because WandbCallback doesn't use epochs as steps
        # important! just make sure to use this callback before WandbCallback
        wandb.log({'predictions': wandb_images}, commit=False)