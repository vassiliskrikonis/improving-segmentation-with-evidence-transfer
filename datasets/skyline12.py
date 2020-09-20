from pathlib import Path
from imageio import imread
import numpy as np
import scipy.io as sio


class Skyline12:
    """
    Skyline12 dataset loader

    example use:
    skyline12 = Skyline12('../input/skylines-12/data')
    img, annot, seed = next(iter(skyline12))
    """

    def __init__(self, root):
        self.root = Path(root)
        self._get_images = lambda: self.root.glob('images/**/*.jpg')

    def __len__(self):
        return len([_ for _ in self._get_images()])

    def __iter__(self):
        images = sorted(list(self._get_images()))
        for path in images:
            image = imread(path)
            city = path.parts[-2]
            annotation = self.root/'annotations'/city/f'label_{path.stem}.mat'
            annotation = sio.loadmat(annotation)['labels']
            seed = self.root/'seeds'/city/f'{path.stem}_fgpixels.mat'
            seed = sio.loadmat(seed)['fgpixels']
            seed = self._seed_as_img(seed, annotation.shape)
            yield image, annotation, seed

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
