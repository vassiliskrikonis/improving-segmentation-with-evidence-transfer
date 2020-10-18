import datetime
from os.path import join
import io
import matplotlib.pyplot as plt
import tensorflow as tf


def get_new_logdir(prefix='', root_dir='.'):
    now = datetime.datetime.now()
    now = '{:%Y-%m-%d %H:%M:%S}'.format(now)
    return join(root_dir, prefix, now)


def plot_to_image(figure):
    """
    copied from:
    https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data

    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it.
    The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
