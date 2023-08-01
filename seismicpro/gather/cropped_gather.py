"""Implements CroppedGather class to crop a Gather and assemble it back from the crops"""

import warnings

import numpy as np

from ..decorators import batch_method


class CroppedGather:
    """A class to construct crops from the input `Gather` data and combine them back into a new `Gather`.

    `CroppedGather` naturally arises during segmentation model training and inference to provide a neural network with
    a batch of fixed-sized inputs and outputs. It can be created by calling `crop` method of the `Gather` class.

    Examples
    --------
    A single crop from a fixed known origin can be obtained as follows:
    >>> crop = gather.crop(origins=(0, 0), crop_shape=(100, 100))

    Cropping is often used during model training to select random regions of a gather and pass them to a model in the
    training loop:
    >>> crops = gather.crop(origins="random", crop_shape=(100, 100), n_crops=16)
    >>> train(model, crops)  # Your train step here

    During inference, you can generate a regular grid of crops covering the whole gather, obtain model predictions for
    them and then aggregate crop predictions back to get a prediction for the whole gather:
    >>> crops = gather.crop(origins="grid", crop_shape=(100, 100), stride=(50, 50))
    >>> predict(model, crops)  # Your inference code here
    >>> gather = crops.assemble_gather()

    Parameters
    ----------
    gather : Gather
        A `Gather` to be cropped.
    origins : 2d np.ndarray
        An array of shape [n_origins, 2] representing absolute coordinates of the first trace of each crop and the
        first time sample respectively.
    crop_shape : tuple with 2 elements
        Shape of the resulting crops.
    pad_mode : str or callable
        Padding mode used when a crop with given origin and shape crosses boundaries of gather data. Passed
        directly to `np.pad`, read https://numpy.org/doc/stable/reference/generated/numpy.pad.html for more
        details.
    kwargs : dict, optional
        Additional keyword arguments to `np.pad`.

    Attributes
    ----------
    gather : Gather
        Cropped `Gather`.
    origins : 2d np.ndarray
        Origins of the crops.
    crop_shape : tuple with 2 elements
        Shape of the crops.
    crops : 3d np.ndarray
        Crops from the gather data, stacked along the zero axis. Has shape (n_crops, *(crop_shape)).
    """
    def __init__(self, gather, origins, crop_shape, pad_mode, **kwargs):
        self.gather = gather
        self.origins = origins
        self.crop_shape = crop_shape
        self.crops = self._make_crops(self._pad_gather(mode=pad_mode, **kwargs))

    @property
    def n_crops(self):
        """int: the number of generated crops."""
        return self.origins.shape[0]

    def _make_crops(self, data):
        """Crop given `data` using known `origins` and `crop_shape`. `data` must be padded so that no crop crosses its
        boundaries."""
        crops = np.empty(shape=(self.n_crops, *self.crop_shape), dtype=data.dtype)
        dx, dy = self.crop_shape
        for i, (x0, y0) in enumerate(self.origins):
            crops[i] = data[x0:x0+dx, y0:y0+dy]
        return crops

    def _pad_gather(self, **kwargs):
        """Pad `gather.data` if needed to perform cropping."""
        max_origins = self.origins.max(axis=0)
        pad_width_x, pad_width_y = np.maximum(0, max_origins + self.crop_shape - self.gather.shape)
        if (pad_width_x > 0) or (pad_width_y > 0):
            warnings.warn("Crop is out of the gather data. The Gather's data will be padded.")
            return np.pad(self.gather.data, ((0, pad_width_x), (0, pad_width_y)), **kwargs)
        return self.gather.data

    @batch_method(target='for', copy_src=False)
    def assemble_gather(self):
        """Assemble crops back to a `Gather` instance.

        The resulting gather will be identical to the gather used to create `self` except for the `data` attribute,
        which will be reconstructed from crops and their origins using mean aggregation. If no crops cover some data
        element, its value will be set to `np.nan` in the resulting gather data.

        Returns
        -------
        gather : Gather
            Assembled gather.
        """
        assembled_data = self._assemble_mean()
        gather = self.gather.copy(ignore='data')
        gather.data = assembled_data
        return gather

    def _assemble_mean(self):
        """Perform mean aggregation of crop data."""
        padded_gather_shape = np.maximum(self.gather.shape, self.crop_shape + self.origins.max(axis=0))
        crops_sum = np.zeros(shape=padded_gather_shape, dtype=np.float32)
        crops_count = np.zeros(shape=padded_gather_shape, dtype=np.int16)
        dx, dy = self.crop_shape
        for crop, (x0, y0) in zip(self.crops, self.origins):
            crops_sum[x0:x0+dx, y0:y0+dy] += crop
            crops_count[x0:x0+dx, y0:y0+dy] += 1
        crops_sum /= crops_count
        return crops_sum[:self.gather.shape[0], :self.gather.shape[1]]
