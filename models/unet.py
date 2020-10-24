from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, \
    Concatenate, Input, Dropout
from tensorflow.keras import Model


def _contracting_block(num_filters, kernel_size, input):
    block_name = f'contracting_block_{num_filters}'
    x = Conv2D(
        num_filters,
        kernel_size,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name=f'{block_name}_conv1'
    )(input)
    skip = Conv2D(
        num_filters,
        kernel_size,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name=f'{block_name}_conv2'
    )(x)
    pool = MaxPool2D(
        2, 2, padding='same', name=f'{block_name}_pool')(skip)
    return pool, skip


def _expanding_block(num_filters, kernel_size, input, skip):
    block_name = f'expanding_block_{num_filters}'
    upsampled = Conv2DTranspose(
        num_filters,
        (2, 2),
        strides=(2, 2),
        padding='same',
        kernel_initializer='he_normal',
        name=f'{block_name}_up'
    )(input)
    x = Concatenate(axis=-1, name=f'{block_name}_concat')([upsampled, skip])
    x = Conv2D(
        num_filters,
        kernel_size,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name=f'{block_name}_conv1'
    )(x)
    x = Conv2D(
        num_filters,
        kernel_size,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name=f'{block_name}_conv2'
    )(x)
    return x


def create_unet(num_classes=5, output_activation='softmax'):
    KERNEL_SIZE = (3, 3)

    img = Input(shape=(None, None, 3), name='input')

    pool1, skip1 = _contracting_block(64, KERNEL_SIZE, img)
    pool2, skip2 = _contracting_block(128, KERNEL_SIZE, pool1)
    pool3, skip3 = _contracting_block(256, KERNEL_SIZE, pool2)
    pool4, skip4 = _contracting_block(512, KERNEL_SIZE, pool3)

    bottom = Dropout(0.3, name='dropout')(pool4)
    bottom = Conv2D(
        1024,
        KERNEL_SIZE,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name='bottom_1024_conv1'
    )(bottom)
    bottom = Conv2D(
        1024,
        KERNEL_SIZE,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name='bottom_1024_conv2'
    )(bottom)

    up4 = _expanding_block(512, KERNEL_SIZE, bottom, skip4)
    up3 = _expanding_block(256, KERNEL_SIZE, up4, skip3)
    up2 = _expanding_block(128, KERNEL_SIZE, up3, skip2)
    up1 = _expanding_block(64, KERNEL_SIZE, up2, skip1)

    out = Conv2D(
        num_classes,
        (1, 1),
        padding='same',
        kernel_initializer='he_normal',
        activation=output_activation,
        name='pixel_classifier'
    )(up1)

    return Model(inputs=img, outputs=out, name='Unet')
