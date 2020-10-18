from pathlib import Path
from imageio import imread
import numpy as np
import scipy.io as sio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from functools import partial


def create_augment_fn(resize_size=512, random_state=42):
    ia.random.seed(random_state)
    augmentor = iaa.Sequential([
        iaa.Sequential([
            iaa.MultiplyBrightness((0.7, 1.5)),
            iaa.MultiplySaturation((0.0, 2.0)),
            iaa.GammaContrast((0.2, 1.4)),
            iaa.Sometimes(1.0, iaa.Crop(percent=(0.1, 0.3))),
            iaa.HorizontalFlip(0.5)
        ], random_order=True),
        iaa.Resize(resize_size)
    ])

    def augment(img, annot, seed, augmentor=augmentor):
        segmap = np.dstack([annot, seed]).astype('uint8')
        aug_img, [aug_segmap] = augmentor.augment(
            image=img, segmentation_maps=[segmap])
        aug_annot, aug_seed = np.dsplit(aug_segmap, 2)
        aug_annot = aug_annot.squeeze()
        aug_seed = aug_seed.squeeze()
        # some labels & scribbles are of bigger size so we need to forcefully
        # resize them to 512x512
        aug_annot = iaa.Resize(
            resize_size, interpolation='nearest').augment_image(aug_annot)
        aug_seed = iaa.Resize(
            resize_size, interpolation='nearest').augment_image(aug_seed)
        return aug_img, aug_annot, aug_seed

    return augment


class Skyline12:
    """
    Skyline12 dataset loader

    example use:
    skyline12 = Skyline12('../input/skylines-12/data')
    img, annot, seed = next(iter(skyline12))
    """

    def __init__(self, root, num_classes=5, resize=512, random_state=42):
        self.root = Path(root)
        self._NUM_CLASSES = num_classes
        self._RESIZE = resize
        self._images = sorted(list(self.root.glob('images/**/*.jpg')))
        self._random_state = random_state
        self._train_set, self._test_set = train_test_split(
            self._images, train_size=100, random_state=random_state)
        self._train_set, self._validation_set = train_test_split(
            self._train_set, train_size=0.8, random_state=random_state + 1)

    def __len__(self):
        return len(self._images)

    def __iter__(self):
        for path in self._images:
            yield self._get_from_path(path)

    def _get_from_path(self, path):
        image = imread(path)
        city = path.parts[-2]
        annotation = self.root / 'annotations' / \
            city / f'label_{path.stem}.mat'
        annotation = sio.loadmat(annotation)['labels']
        seed = self.root / 'seeds' / city / f'{path.stem}_fgpixels.mat'
        seed = sio.loadmat(seed)['fgpixels']
        seed = self._seed_as_img(seed, annotation.shape)
        return image, annotation, seed

    def _seed_as_img(self, seed, shape, stroke_size=6):
        buildings = seed[0]
        mask = np.zeros(shape)
        for idx, points in enumerate(buildings):
            for [x, y] in points:
                x_min = max(0, x - stroke_size // 2)
                y_min = max(0, y - stroke_size // 2)
                x_max = min(shape[1], x + stroke_size)
                y_max = min(shape[0], y + stroke_size)
                mask[y_min:y_max, x_min:x_max] = idx
        return mask

    def as_tf_dataset(
        self, folds=2, subset=None, keep_individual=False, cache_dir=None
    ):
        """
        Returns Skyline12 as a tf.data.Dataset.
        Can specify which subset (training, validation, test)
        to return
        """
        samples = self._images
        if subset == 'train':
            samples = self._train_set
        elif subset == 'validation':
            samples = self._validation_set
        elif subset == 'test':
            samples = self._test_set

        augment = create_augment_fn(random_state=self._random_state)

        def ds_gen(sample_set, folds):
            for _ in range(folds):
                for x, y, z in (self._get_from_path(p) for p in sample_set):
                    yield self.preprocess(x, y, z, augment, keep_individual)

        ds = tf.data.Dataset.from_generator(
            partial(ds_gen, samples, folds),
            (
                tf.dtypes.float32,
                tf.dtypes.uint8,
                tf.dtypes.float32
            ),
            (
                tf.TensorShape([512, 512, 3]),
                tf.TensorShape([512, 512, self._NUM_CLASSES]),
                tf.TensorShape([512, 512, 1])
            )
        )
        if cache_dir:
            cache = Path(cache_dir) / (
                f'{subset if subset else "all"}'
                f'_folds{folds}'
                f'_{"squashed" if not keep_individual else ""}'
            )
            ds = ds.cache(str(cache))
        return ds

    def preprocess(self, x, y, z, augment_fn, keep_individual=False):
        x, y, z = augment_fn(x, y, z)
        x = x.astype('float32')
        x /= 255.0
        if not keep_individual:
            y[y >= self._NUM_CLASSES - 1] = self._NUM_CLASSES - 1
        y = tf.keras.utils.to_categorical(y, num_classes=self._NUM_CLASSES)
        z = z.astype('float32')
        if not keep_individual:
            z[z > 0] = 1.0
        z = np.expand_dims(z, -1)
        return x, y, z

    @staticmethod
    def show_sample(image, masks=[], from_tensors=False):
        rows = 1
        cols = 1 + len(masks)
        fig, axes = plt.subplots(rows, cols)
        fig.set_size_inches((10, 4))
        if rows * cols == 1:
            axes.imshow(image)
        else:
            axes[0].imshow(image)
        for i, mask in enumerate(masks):
            if from_tensors:
                mask = mask.numpy()
            if len(mask.shape) > 2:
                if mask.shape[-1] == 1:
                    pass
                    # mask = mask.squeeze(-1)
                else:
                    mask = mask.argmax(-1)
            else:
                mask = np.expand_dims(mask, -1)
            axes[1 + i].imshow(mask)
        return fig
