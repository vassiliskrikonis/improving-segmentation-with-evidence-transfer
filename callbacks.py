import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import numpy as np
from itertools import chain


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
        wandb.log({'predictions': wandb_images}, step=epoch)


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
        wandb.log({'predictions': wandb_images}, step=epoch)


class MyWandbCallback(WandbCallback):
    def __init__(self, include='xyz', val_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include = include
        self.val_data = val_data

    def _log_images(self, num_images=36):
        validation_X = (self.val_data[0]
                        if self.val_data
                        else self.validation_data[0])
        validation_y, validation_z = (self.val_data[1]
                                      if self.val_data
                                      else self.validation_data[1])

        validation_length = len(validation_X)

        if validation_length > num_images:
            # pick some data at random
            indices = np.random.choice(validation_length, num_images, replace=False)
        else:
            indices = range(validation_length)

        test_data = []
        test_output_y = []
        test_output_z = []

        for i in indices:
            test_example = validation_X[i]
            test_data.append(test_example)
            test_output_y.append(validation_y[i])
            test_output_z.append(validation_z[i])

        if self.model.stateful:
            [predictions_y, predictions_z] = self.model.predict(np.stack(test_data), batch_size=1)
            self.model.reset_states()
        else:
            if not hasattr(self, '_prediction_batch_size'):
                self._prediction_batch_size = 1
            [predictions_y, predictions_z] = self.model.predict(
                np.stack(test_data), batch_size=self._prediction_batch_size
            )
            if len(predictions_y) != len(test_data):
                self._prediction_batch_size = 1
                [predictions_y, predictions_z] = self.model.predict(
                    np.stack(test_data), batch_size=self._prediction_batch_size
                )

        input_image_data = test_data
        output_image_data_y = self._masks_to_pixels(predictions_y)
        output_image_data_z = self._masks_to_pixels(predictions_z)
        reference_image_data_y = self._masks_to_pixels(np.stack(test_output_y))
        reference_image_data_z = self._masks_to_pixels(np.stack(test_output_z))
        input_images = [
            wandb.Image(data, grouping=len(self.include))
            for i, data in enumerate(input_image_data)
        ]
        output_images_y = [
            wandb.Image(data) for i, data in enumerate(output_image_data_y)
        ]
        output_images_z = [
            wandb.Image(data) for i, data in enumerate(output_image_data_z)
        ]
        reference_images_y = [
            wandb.Image(data) for i, data in enumerate(reference_image_data_y)
        ]
        reference_images_z = [
            wandb.Image(data) for i, data in enumerate(reference_image_data_z)
        ]
        for_logging = []
        if 'x' in self.include:
            for_logging.append(input_images)
        if 'y' in self.include:
            for_logging.append(output_images_y)
        if 'z' in self.include:
            for_logging.append(output_images_z)
        if 'y' in self.include:
            for_logging.append(reference_images_y)
        if 'z' in self.include:
            for_logging.append(reference_images_z)
        return list(chain.from_iterable(zip(*for_logging)))