from utils import plot_to_image
from datasets.skyline12 import Skyline12
import tensorflow as tf
import wandb


class LogImages(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, pred_data):
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.pred_data = pred_data

    def on_epoch_end(self, epoch, logs=None):
        xs, ys = self.pred_data
        preds = self.model(xs, training=False)
        with self.writer.as_default():
            figs = []
            for x, y, pred_y in zip(xs, ys, preds):
                fig = Skyline12.show_sample(x, [y, pred_y], from_tensors=True)
                fig = plot_to_image(fig)
                # remove the batch dimensions added in plot_to_image, since
                # we add multiple images after all
                fig = tf.squeeze(fig, axis=0)
                figs.append(fig)
            tf.summary.image(
                'Predictions', figs, max_outputs=len(preds), step=epoch)


class LogImagesWandb(tf.keras.callbacks.Callback):
    def __init__(self, pred_data):
        self.pred_data = pred_data

    def on_epoch_end(self, epoch, logs=None):
        xs, ys = self.pred_data
        preds = self.model(xs, training=False)
        figs = []
        for x, y, pred_y in zip(xs, ys, preds):
            fig = Skyline12.show_sample(x, [y, pred_y], from_tensors=True)
            fig = plot_to_image(fig)
            # remove the batch dimensions added in plot_to_image, since
            # we add multiple images after all
            fig = tf.squeeze(fig, axis=0)
            figs.append(fig)
        wandb.log(
            {'predictions': [wandb.Image(fig) for fig in figs]}, step=epoch)


class LogEviTRAMImagesWandb(tf.keras.callbacks.Callback):
    def __init__(self, pred_data):
        self.pred_data = pred_data

    def on_epoch_end(self, epoch, logs=None):
        xs, [ys, zs] = self.pred_data
        preds = self.model(xs, training=False)
        figs = []
        for x, y, z, pred_y, pred_z in zip(xs, ys, zs, *preds):
            fig = Skyline12.show_sample(
                x, [y, z, pred_y, pred_z], from_tensors=True)
            fig = plot_to_image(fig)
            # remove the batch dimensions added in plot_to_image, since
            # we add multiple images after all
            fig = tf.squeeze(fig, axis=0)
            figs.append(fig)
        wandb.log(
            {'predictions': [wandb.Image(fig) for fig in figs]}, step=epoch)
