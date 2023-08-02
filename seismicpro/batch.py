"""Implements SeismicBatch class for processing a small subset of seismic gathers"""

from string import Formatter
from functools import partial
from itertools import zip_longest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from batchflow import save_data_to, Batch, DatasetIndex, NamedExpression
from batchflow.decorators import action, inbatch_parallel

from .config import config
from .index import SeismicIndex
from .gather import Gather, CroppedGather
from .gather.utils.crop_utils import make_origins
from .velocity_spectrum import VerticalVelocitySpectrum, ResidualVelocitySpectrum
from .field import Field
from .metrics import define_pipeline_metric
from .decorators import create_batch_methods, apply_to_each_component
from .utils import to_list, align_src_dst, as_dict, save_figure


@create_batch_methods(Gather, CroppedGather, VerticalVelocitySpectrum, ResidualVelocitySpectrum)
class SeismicBatch(Batch):
    """A batch class for seismic data that allows for joint and simultaneous processing of small subsets of seismic
    gathers in a parallel way.

    Initially, a batch contains unique identifiers of seismic gathers as its `index` and allows for their loading and
    processing. All the results are stored in batch attributes called `components` whose names are passed as `dst`
    argument of the called method.

    `SeismicBatch` implements almost no processing logic itself and usually just redirects method calls to objects in
    components specified in `src` argument. In order for a component method to be available in the batch, it should be
    decorated with :func:`~decorators.batch_method` in its class and the class itself should be listed in
    :func:`~decorators.create_batch_methods` decorator arguments of `SeismicBatch`.

    Examples
    --------
    Usually a batch is created from a `SeismicDataset` instance by calling :func:`~SeismicDataset.next_batch` method:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
    >>> dataset = SeismicDataset(survey)
    >>> batch = dataset.next_batch(10)

    Here a batch of 10 gathers was created and can now be processed using the methods defined in
    :class:`~batch.SeismicBatch`. The batch does not contain any data yet and gather loading is usually the first
    method you want to call:
    >>> batch.load(src="survey")

    We've loaded gathers from a survey called `survey` in the component with the same name. Now the data can be
    accessed as a usual attribute:
    >>> batch.survey

    Almost all methods return a transformed batch allowing for method chaining:
    >>> batch.sort(src="survey", by="offset").plot(src="survey")

    Note that if `dst` attribute is omitted data processing is performed inplace.

    Parameters
    ----------
    index : DatasetIndex
        Unique identifiers of seismic gathers in the batch. Usually has :class:`~index.SeismicIndex` type.

    Attributes
    ----------
    index : DatasetIndex
        Unique identifiers of seismic gathers in the batch. Usually has :class:`~index.SeismicIndex` type.
    components : tuple of str or None
        Names of the created components. Each of them can be accessed as a usual attribute.
    is_combined : bool or None
        Whether gathers in the batch are combined. `None` until `load` method is called for the first time.
    initial_index : SeismicIndex
        An index used to create the batch.
    n_calculated_metrics : int
        The number of times `calculate_metric` method was called for the batch.
    """
    def __init__(self, *args, is_combined=None, initial_index=None, n_calculated_metrics=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_combined = is_combined
        self.initial_index = self.index if initial_index is None else initial_index
        self.n_calculated_metrics = n_calculated_metrics

    def __getstate__(self):
        """Create pickling state of a batch from its `__dict__`. Don't pickle `dataset` and `pipeline` if
        `enable_fast_pickling` config option is set."""
        state = super().__getstate__()
        if config["enable_fast_pickling"]:
            state["_dataset"] = None
            state["pipeline"] = None
        return state

    def init_component(self, *args, dst=None, **kwargs):
        """Create and preallocate new attributes with names listed in `dst` if they don't exist and return ordinal
        numbers of batch items. This method is typically used as a default `init` function in `inbatch_parallel`
        decorator."""
        _ = args, kwargs
        dst = [] if dst is None else to_list(dst)
        for comp in dst:
            if self.components is None or comp not in self.components:
                self.add_components(comp, init=self.array_of_nones)
        return np.arange(len(self))

    def init_coef_component(self, *args, dst=None, dst_coefs=None, **kwargs):
        """Create and preallocate new attributes with names listed in `dst` and `dst_coefs` if they don't exist and
        return ordinal numbers of batch items. This method is used as a default `init` for `apply_agc` method."""
        dst_coefs = [] if dst_coefs is None else to_list(dst_coefs)
        dst = [] if dst is None else to_list(dst)
        return self.init_component(*args, dst=dst+dst_coefs, **kwargs)

    @property
    def flat_indices(self):
        """np.ndarray: Unique identifiers of seismic gathers in the batch flattened into a 1d array."""
        if isinstance(self.index, SeismicIndex):
            return np.concatenate(self.indices)
        return self.indices

    @action
    def load(self, src=None, dst=None, fmt="sgy", combined=False, limits=None, copy_headers=False, chunk_size=1000,
             n_workers=None, **kwargs):
        """Load seismic gathers into batch components.

        Parameters
        ----------
        src : str or list of str, optional
            Survey names to load gathers from.
        dst : str or list of str, optional
            Batch components to store the result in. Equals to `src` if not given.
        fmt : str, optional, defaults to "sgy"
            Data format to load gathers from.
        combined : bool, optional, defaults to False
            If `False`, load gathers by corresponding index value. If `True`, group all traces from a particular survey
            into a single gather. Increases loading speed by reducing the number of `DataFrame` indexations performed.
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to `__init__` of the corresponding survey are used. Measured in samples.
        copy_headers : bool, optional, defaults to False
            Whether to copy the subset of survey `headers` describing the gather.
        chunk_size : int, optional, defaults to 1000
            The number of traces to load by each of spawned threads.
        n_workers : int, optional
            The maximum number of simultaneously spawned threads to load traces. Defaults to the number of cpu cores.
        kwargs : misc, optional
            Additional keyword arguments to be passed to `super().load` if `fmt` is not "sgy" or "segy".

        Returns
        -------
        batch : SeismicBatch
            A batch with loaded gathers. Creates or updates `dst` components inplace.

        Raises
        ------
        KeyError
            If unknown survey name was passed in `src`.
        """
        if not isinstance(fmt, str) or fmt.lower() not in {"sgy", "segy"}:
            return super().load(src=src, fmt=fmt, dst=dst, **kwargs)

        if self.is_combined is not None and combined != self.is_combined:
            raise ValueError("combined flag must match the batch combined status")

        src, dst = align_src_dst(src, dst)
        non_empty_parts = [i for i, n_gathers in enumerate(self.initial_index.n_gathers_by_part) if n_gathers]
        batch = type(self)(DatasetIndex(non_empty_parts), dataset=self.dataset, pipeline=self.pipeline,
                           is_combined=True, initial_index=self.initial_index,
                           n_calculated_metrics=self.n_calculated_metrics)
        batch = batch.load_combined_gather(src=src, dst=dst, limits=limits, copy_headers=copy_headers,
                                           chunk_size=chunk_size, n_workers=n_workers)
        if not combined:
            batch = batch.split_gathers(src=dst, assume_sequential=True)

        if self.is_combined is None:  # the first data loading into the batch
            return batch

        for component in dst:  # copy loaded components to self
            component_data = getattr(batch, component)
            if hasattr(self, component):
                setattr(self, component, component_data)
            else:
                self.add_components(component, component_data)
        return self

    @apply_to_each_component(target="for", fetch_method_target=False)
    def load_combined_gather(self, pos, src, dst, limits=None, copy_headers=False, chunk_size=1000, n_workers=None):
        """Load all batch traces from a given part and survey into a single gather."""
        part = self.initial_index.parts[self.indices[pos]]
        survey = part.surveys_dict[src]
        headers = part.headers.get(src, part.headers[[]])  # Handle the case when no headers were loaded for a survey
        getattr(self, dst)[pos] = survey.load_gather(headers, limits=limits, copy_headers=copy_headers,
                                                     chunk_size=chunk_size, n_workers=n_workers)

    @action
    def combine_gathers(self, src, dst=None):
        """Combine all gathers produced by the same survey into a single gather for each component in `src`.

        Most tracewise actions benefit from processing large gathers at once. Combining individual gathers into a
        single one, running all tracewise methods and splitting the resulting gather back may significantly speed up
        the processing pipeline.

        This method also allows for more efficient subsequent gather `dump` since:
        1. Text and bin headers are stored only once for the whole combined gather, which especially matters when
           dumping a single stacked trace,
        2. Total number of gathers passed to `aggregate_segys` is reduced.

        Parameters
        ----------
        src : str or list of str
            Batch components to combine.
        dst : str or list of str, optional
            Batch components to store the combined gathers in. Equals to `src` if not given.

        Returns
        -------
        batch : SeismicBatch
            A batch with combined gathers.
        """
        if self.is_combined is None or self.is_combined:
            return self

        src_list, dst_list = align_src_dst(src, dst)
        non_empty_parts = [i for i, n_gathers in enumerate(self.index.n_gathers_by_part) if n_gathers]
        combined_batch = type(self)(DatasetIndex(non_empty_parts), dataset=self.dataset, pipeline=self.pipeline,
                                    is_combined=True, initial_index=self.initial_index,
                                    n_calculated_metrics=self.n_calculated_metrics)
        split_pos = np.cumsum([n_gathers for n_gathers in self.index.n_gathers_by_part if n_gathers][:-1])
        for src, dst in zip(src_list, dst_list):  # pylint: disable=redefined-argument-from-local
            gathers = getattr(self, src)
            if not all(isinstance(gather, Gather) for gather in gathers):
                raise ValueError(f"{src} component contains items that are not instances of Gather class")
            combined_gathers = []
            for gather_chunk in np.split(gathers, split_pos):
                samples_params = {(gather.n_times, gather.sample_interval, gather.delay) for gather in gather_chunk}
                if len(samples_params) != 1:
                    raise ValueError(f"All gathers in {src} component must have the same samples")
                _, sample_interval, delay = samples_params.pop()
                headers = pd.concat([gather.headers for gather in gather_chunk])
                data = np.concatenate([gather.data for gather in gather_chunk])
                gather = Gather(headers, data, sample_interval=sample_interval, delay=delay,
                                survey=gather_chunk[0].survey)
                combined_gathers.append(gather)
            combined_batch.add_components(dst, init=np.array(combined_gathers))
        return combined_batch

    @action
    def split_gathers(self, src, dst=None, assume_sequential=True):
        """Split combined gathers in each component in `src` and store them in the corresponding components of `dst`.

        Most tracewise actions benefit from processing large gathers at once. Combining individual gathers into a
        single one, running all tracewise methods and splitting the resulting gather back may significantly speed up
        the processing pipeline.

        Parameters
        ----------
        src : str or list of str
            Batch components to split.
        dst : str or list of str, optional
            Batch components to store the split gathers in. Equals to `src` if not given.
        assume_sequential : bool, optional, defaults to True
            Assume that individual gathers follow one another in the combined ones. If `True`, avoids excessive copies
            and allows for much more efficient splitting.

        Returns
        -------
        batch : SeismicBatch
            A batch with split gathers.
        """
        if self.is_combined is None or not self.is_combined:
            return self

        src_list, dst_list = align_src_dst(src, dst)
        split_batch = type(self)(self.initial_index, dataset=self.dataset, pipeline=self.pipeline,
                                 is_combined=False, initial_index=self.initial_index,
                                 n_calculated_metrics=self.n_calculated_metrics)
        for src, dst in zip(src_list, dst_list):  # pylint: disable=redefined-argument-from-local
            gathers = getattr(self, src)
            if not all(isinstance(gather, Gather) for gather in gathers):
                raise ValueError(f"{src} component contains items that are not instances of Gather class")

            gather_indices = []
            if assume_sequential:
                for gather in gathers:
                    split_pos = np.where(~gather.headers.index.duplicated(keep="first"))[0]
                    gather_indices.append([slice(start, end) for start, end in zip_longest(split_pos, split_pos[1:])])
            else:
                for gather in gathers:
                    gather_indices.append(gather.headers.groupby(gather.indexed_by, sort=False).indices.values())
            split_gathers = [gather[ix] for gather, indices in zip(gathers, gather_indices) for ix in indices]
            split_batch.add_components(dst, init=np.array(split_gathers))
        return split_batch

    @action
    def update_field(self, field, src):
        """Update a field with objects from `src` component.

        Parameters
        ----------
        field : Field
            A field to update.
        src : str
            A component of instances to update the cube with. Each of them must have well-defined coordinates.

        Returns
        -------
        self : SeismicBatch
            The batch unchanged.
        """
        if not isinstance(field, Field):
            raise ValueError("Only a Field instance can be updated")
        field.update(getattr(self, src))
        return self

    @action
    def make_model_inputs(self, src, dst, mode='c', axis=0, expand_dims_axis=None):
        """Transform data to be used for model training.

        The method performs two-stage data processing:
        1. Stacks or concatenates input data depending on `mode` parameter along the specified `axis`,
        2. Inserts new axes to the resulting array at positions specified by `expand_dims_axis`.

        Source data to be transformed is passed to `src` argument either as an array-like of `np.ndarray`s or as a
        string, representing a name of batch component to get data from. Since this method is usually called in model
        training pipelines, `BA` named expression can be used to extract a certain attribute from each element of given
        component.

        Examples
        --------
        Given a dataset of individual traces, extract them from a batch of size 3 using `BA` named expression,
        concatenate into a single array, add a dummy axis and save the result into the `inputs` component:
        >>> pipeline = (Pipeline()
        ...     .load(src='survey')
        ...     .make_model_inputs(src=L('survey').data, dst='inputs', mode='c', axis=0, expand_dims_axis=1)
        ... )
        >>> batch = (dataset >> pipeline).next_batch(3)
        >>> batch.inputs.shape
        (3, 1, 1500)

        Parameters
        ----------
        src : src or array-like of np.ndarray
            Either a data to be processed itself or a component name to get it from.
        dst : src
            A component's name to store the combined result in.
        mode : {'c' or 's'}, optional, defaults to 'c'
            A mode that determines how to combine a sequence of arrays into a single one: 'c' stands for concatenating
            and 's' for stacking along the `axis`.
        axis : int or None, optional, defaults to 0
            An axis along which the arrays will be concatenated or stacked. If `mode` is `c`, `None` can be passed
            meaning that the arrays will be flattened before concatenation. Regardless of `mode`, `axis` must be no
            more than `data.ndim` - 1.
        expand_dims_axis : int or None, optional, defaults to None
            Insert new axes at the `expand_dims_axis` position in the expanded array. If `None`, the expansion does not
            occur.

        Returns
        -------
        self : SeismicBatch
            Batch with the resulting `np.ndarray` in the `dst` component.

        Raises
        ------
        ValueError
            If unknown `mode` was passed.
        """
        data = getattr(self, src) if isinstance(src, str) else src
        func = {'c': np.concatenate, 's': np.stack}.get(mode)
        if func is None:
            raise ValueError(f"Unknown mode '{mode}', must be either 'c' or 's'")
        data = func(data, axis=axis)

        if expand_dims_axis is not None:
            data = np.expand_dims(data, axis=expand_dims_axis)
        setattr(self, dst, data)
        return self

    @action(no_eval='dst')
    def split_model_outputs(self, src, dst, shapes):
        """Split data into multiple sub-arrays whose shapes along zero axis are defined by `shapes`.

        Usually gather data for each batch element is stacked or concatenated along zero axis using
        :func:`SeismicBatch.make_model_inputs` before being passed to a model. This method performs a reverse operation
        by splitting the received predictions allowing them to be matched with the corresponding batch elements for
        which they were obtained.

        Examples
        --------
        Given a dataset of individual traces, perform a segmentation model inference for a batch of size 3, split
        predictions and save them to the `outputs` batch component:
        >>> pipeline = (Pipeline()
        ...     .init_model(mode='dynamic', model_class=UNet, name='model', config=config)
        ...     .init_variable('predictions')
        ...     .load(src='survey')
        ...     .make_model_inputs(src=L('survey').data, dst='inputs', mode='c', axis=0, expand_dims_axis=1)
        ...     .predict_model('model', B('inputs'), fetches='predictions', save_to=B('predictions'))
        ...     .split_model_outputs(src='predictions', dst='outputs', shapes=L('survey').shape[0])
        ... )
        >>> batch = (dataset >> pipeline).next_batch(3)

        Each gather in the batch has shape (1, 1500), thus the created model inputs have shape (3, 1, 1500). Model
        predictions have the same shape as inputs:
        >>> batch.inputs.shape
        (3, 1, 1500)
        >>> batch.predictions.shape
        (3, 1, 1500)

        Predictions are split into 3 subarrays with a single trace in each of them to match the number of traces in the
        corresponding gathers:
        >>> len(batch.outputs)
        3
        >>> batch.outputs[0].shape
        (1, 1, 1500)

        Parameters
        ----------
        src : str or array-like of np.ndarray
            Either a data to be processed itself or a component name to get it from.
        dst : str or NamedExpression
            - If `str`, save the resulting sub-arrays into a batch component called `dst`,
            - If `NamedExpression`, save the resulting sub-arrays into the object described by named expression.
        shapes : 1d array-like
            An array with sizes of each sub-array along zero axis after the split. Its length should be generally equal
            to the current batch size and its sum must match the length of data defined by `src`.

        Returns
        -------
        self : SeismicBatch
            The batch with split data.

        Raises
        ------
        ValueError
            If data length does not match the sum of shapes passed.
            If `dst` is not of `str` or `NamedExpression` type.
        """
        data = getattr(self, src) if isinstance(src, str) else src
        shapes = np.cumsum(shapes)
        if shapes[-1] != len(data):
            raise ValueError("Data length must match the sum of shapes passed")
        split_data = np.split(data, shapes[:-1])

        if isinstance(dst, str):
            setattr(self, dst, split_data)
        elif isinstance(dst, NamedExpression):
            dst.set(value=split_data)
        else:
            raise ValueError(f"dst must be either `str` or `NamedExpression`, not {type(dst)}.")
        return self

    @action
    @inbatch_parallel(init="init_component", target="threads")
    def crop(self, pos, src, origins, crop_shape, dst=None, joint=True, n_crops=1, stride=None, **kwargs):
        """Crop batch components.

        Parameters
        ----------
        src : str or list of str
            Components to be cropped. Objects in each of them must implement `crop` method which will be called from
            this method.
        origins : list, tuple, np.ndarray or str
            Origins define top-left corners for each crop or a rule used to calculate them. All array-like values are
            cast to an `np.ndarray` and treated as origins directly, except for a 2-element tuple of `int`, which will
            be treated as a single individual origin.
            If `str`, represents a mode to calculate origins. Two options are supported:
            - "random": calculate `n_crops` crops selected randomly using a uniform distribution over the source data,
              so that no crop crosses data boundaries,
            - "grid": calculate a deterministic uniform grid of origins, whose density is determined by `stride`.
        crop_shape : tuple with 2 elements
            Shape of the resulting crops.
        dst : str or list of str, optional, defaults to None
            Components to store cropped data. If `dst` is `None` cropping is performed inplace.
        joint : bool, optional, defaults to True
            Defines whether to create the same origins for all `src`s if passed `origins` is `str`. Generally used to
            perform joint random cropping of segmentation model input and output.
        n_crops : int, optional, defaults to 1
            The number of generated crops if `origins` is "random".
        stride : tuple with 2 elements, optional, defaults to crop_shape
            Steps between two adjacent crops along both axes if `origins` is "grid". The lower the value is, the more
            dense the grid of crops will be. An extra origin will always be placed so that the corresponding crop will
            fit in the very end of an axis to guarantee complete data coverage with crops regardless of passed
            `crop_shape` and `stride`.
        kwargs : misc, optional
            Additional keyword arguments to pass to `crop` method of the objects being cropped.

        Returns
        -------
        self : SeismicBatch
            The batch with cropped data.

        Raises
        ------
        TypeError
            If `joint` is `True` and `src` contains components of different types.
        ValueError
            If `src` and `dst` have different lengths.
            If `joint` is `True` and `src` contains components of different shapes.
        """
        src_list, dst_list = align_src_dst(src, dst)

        if joint:
            src_shapes = set()
            src_types = set()

            for src in src_list:  # pylint: disable=redefined-argument-from-local
                src_obj = getattr(self, src)[pos]
                src_types.add(type(src_obj))
                src_shapes.add(src_obj.shape)

            if len(src_types) > 1:
                raise TypeError("If joint is True, all src components must be of the same type.")
            if len(src_shapes) > 1:
                raise ValueError("If joint is True, all src components must have the same shape.")
            data_shape = src_shapes.pop()
            origins = make_origins(origins, data_shape, crop_shape, n_crops, stride)

        for src, dst in zip(src_list, dst_list):  # pylint: disable=redefined-argument-from-local
            src_obj = getattr(self, src)[pos]
            src_cropped = src_obj.crop(origins, crop_shape, n_crops, stride, **kwargs)
            setattr(self[pos], dst, src_cropped)

        return self

    @action
    @inbatch_parallel(init="init_coef_component", target="threads")
    def apply_agc(self, pos, src, dst=None, dst_coefs=None, window_size=250, mode='rms'):
        """Calculate instantaneous or RMS amplitude AGC coefficients and apply them to gather data.

        Parameters
        ----------
        src : str or list of str
            Batch components with gather instances to apply AGC to.
        dst : str or list of str, optional
            Batch components to store the scaled gathers in. Equals to `src` if not given.
        dst_coefs : str or list of str, optional
            Batch components to store AGC coefficients in.
        window_size : int, optional, defaults to 250
            Window size to calculate AGC scaling coefficient in, measured in milliseconds.
        mode : str, optional, defaults to 'rms'
            Mode for AGC: if 'rms', root mean squared value of non-zero amplitudes in the given window
            is used as scaling coefficient (RMS amplitude AGC), if 'abs' - mean of absolute non-zero
            amplitudes (instantaneous AGC).

        Returns
        -------
        self : SeismicBatch
            The batch with scaled gathers and optionally AGC coefficients.
        """
        src_list, dst_list = align_src_dst(src, dst)
        dst_coefs_list = to_list(dst_coefs) if dst_coefs is not None else [None] * len(dst_list)
        if len(dst_coefs_list) != len(dst_list):
            raise ValueError("dst_coefs and dst should have the same length.")

        # pylint: disable-next=redefined-argument-from-local
        for src, dst_coef, dst in zip(src_list, dst_coefs_list, dst_list):
            src_obj = getattr(self, src)[pos]
            src_obj = src_obj.copy() if src != dst else src_obj
            return_coefs = dst_coef is not None
            results = src_obj.apply_agc(window_size=window_size, mode=mode, return_coefs=return_coefs)
            if return_coefs:
                setattr(self[pos], dst, results[0])
                setattr(self[pos], dst_coef, results[1])
            else:
                setattr(self[pos], dst, results)
        return self

    @action
    @inbatch_parallel(init="init_component", target="threads")
    def undo_agc(self, pos, src, src_coefs, dst=None):
        """Undo previously applied AGC correction using precomputed AGC coefficients.

        Parameters
        ----------
        src : str or list of str
            Batch components with scaled gather instances.
        src_coefs : str or list of str
            Batch components with gather instances with AGC coefficients in the `data` attribute.
        dst : str or list of str, optional
            Batch components to store unscaled gathers in. Equals to `src` if not given.

        Returns
        -------
        self : SeismicBatch
            The batch with unscaled gathers.

        Raises
        ------
        ValueError
            If `src` and `src_coefs` have different lengths.
        """
        src_list, dst_list = align_src_dst(src, dst)
        src_coefs_list = to_list(src_coefs)

        if len(src_list) != len(src_coefs_list):
            raise ValueError("The length of `src_coefs` must match the length of `src`")

        # pylint: disable-next=redefined-argument-from-local
        for src, coef, dst in zip(src_list, src_coefs_list, dst_list):
            src_obj = getattr(self, src)[pos]
            src_coef = getattr(self, coef)[pos]

            src_obj = src_obj.copy() if src != dst else src_obj
            src_noagc = src_obj.undo_agc(coefs_gather=src_coef)
            setattr(self[pos], dst, src_noagc)
        return self

    @action(no_eval="save_to")
    def calculate_metric(self, metric, *args, metric_name=None, coords_component=None, save_to=None, **kwargs):
        """Calculate a metric for each batch element and store the results into a metric map.

        The passed metric must be either an instance or a subclass of `PipelineMetric` or a `callable`. In the latter
        case, a new instance of `FunctionalMetric` will be created with its `__call__` method defined by the callable.
        The metric is provided with information about the pipeline it was calculated in, which allows restoring a batch
        of data by its spatial coordinates upon interactive metric map plotting.

        Examples
        --------
        1. Calculate a metric, that estimates signal leakage after seismic processing by CDP gathers:

        Create a dataset with surveys before and after processing being merged:
        >>> header_index = ["INLINE_3D", "CROSSLINE_3D"]
        >>> header_cols = "offset"
        >>> survey_before = Survey(path_before, header_index=header_index, header_cols=header_cols, name="before")
        >>> survey_after = Survey(path_after, header_index=header_index, header_cols=header_cols, name="after")
        >>> dataset = SeismicDataset(survey_before, survey_after, mode="m")

        Iterate over the dataset and calculate the metric:
        >>> pipeline = (dataset
        ...     .pipeline()
        ...     .load(src=["before", "after"])
        ...     .calculate_metric(SignalLeakage, "before", "after", velocities=np.linspace(1500, 5500, 100),
        ...                       save_to=V("map", mode="a"))
        ... )
        >>> pipeline.run(batch_size=16, n_epochs=1)

        Extract the created map and plot it:
        >>> leakage_map = pipeline.v("map")
        >>> leakage_map.plot(interactive=True)  # works only in JupyterLab with `%matplotlib widget` magic executed

        2. Calculate standard deviation of gather amplitudes using a lambda-function:
        >>> pipeline = (dataset
        ...     .pipeline()
        ...     .load(src="before")
        ...     .calculate_metric(lambda gather: gather.data.std(), "before", metric_name="std",
        ...                       save_to=V("map", mode="a"))
        ... )
        >>> pipeline.run(batch_size=16, n_epochs=1)
        >>> std_map = pipeline.v("map")
        >>> std_map.plot(interactive=True, plot_component="before")

        Parameters
        ----------
        metric : PipelineMetric or subclass of PipelineMetric or callable
            The metric to calculate.
        metric_name : str, optional
            A name of the metric.
        coords_component : str, optional
            A component name to extract coordinates from. If not given, the first argument passed to the metric
            calculation function is used.
        save_to : NamedExpression
            A named expression to save the constructed `MetricMap` instance to.
        args : misc, optional
            Additional positional arguments to the metric calculation function.
        kwargs : misc, optional
            Additional keyword arguments to the metric calculation function.

        Returns
        -------
        self : SeismicBatch
            The batch with increased `n_calculated_metrics` counter.

        Raises
        ------
        TypeError
            If wrong type of `metric` is passed.
        ValueError
            If some batch item has `None` coordinates.
        """
        metric = define_pipeline_metric(metric, metric_name)
        unpacked_args, first_arg = metric.unpack_calc_args(self, *args, **kwargs)

        # Calculate metric values and their coordinates
        values = [metric(*args, **kwargs) for args, kwargs in unpacked_args]
        coords_items = first_arg if coords_component is None else getattr(self, coords_component)
        coords = [item.coords for item in coords_items]
        if None in coords:
            raise ValueError("All batch items must have well-defined coordinates")

        # Construct metric map index as a concatenation of dataset part and batch index
        part_indices = []
        for i, ix in enumerate(self.indices):
            if len(ix):
                ix = ix.to_frame(index=False)
                ix.insert(0, "Part", i)
                part_indices.append(ix)
        index = pd.concat(part_indices, ignore_index=True, copy=False)

        # Construct and save the map
        metric = metric.provide_context(pipeline=self.pipeline, calculate_metric_index=self.n_calculated_metrics)
        metric_map = metric.construct_map(coords, values, index=index, calculate_immediately=False)
        if save_to is not None:
            save_data_to(data=metric_map, dst=save_to, batch=self)
        self.n_calculated_metrics += 1
        return self

    @staticmethod
    def _unpack_args(args, batch_item):
        """Replace all names of batch components in `args` with corresponding values from `batch_item`. """
        if not isinstance(args, (list, tuple, str)):
            return args

        unpacked_args = [getattr(batch_item, val) if isinstance(val, str) and val in batch_item.components else val
                         for val in to_list(args)]
        if isinstance(args, str):
            return unpacked_args[0]
        return unpacked_args

    @action  # pylint: disable-next=too-many-statements
    def plot(self, src, src_kwargs=None, max_width=20, title="{src}: {index}", save_to=None, **common_kwargs):
        """Plot batch components on a grid constructed as follows:
        1. If a single batch component is passed, its objects are plotted side by side on a single line.
        2. Otherwise, each batch element is drawn on a separate line, its components are plotted in the order they
           appear in `src`.

        If the total width of plots on a line exceeds `max_width`, the line is wrapped and the plots that did not fit
        are drawn below.

        This action calls `plot` methods of objects in components in `src`. There are two ways to pass arguments to
        these methods:
        1. `common_kwargs` set defaults for all of them,
        2. `src_kwargs` define specific `kwargs` for an individual component that override those in `common_kwargs`.

        Notes
        -----
        1. `kwargs` from `src_kwargs` take priority over the `common_kwargs` and `title` argument.
        2. `title` is processed differently than in the `plot` methods of objects in `src` components, see its
           description below for more details.

        Parameters
        ----------
        src : str or list of str
            Components to be plotted. Objects in each of them must implement `plot` method which will be called from
            this method.
        src_kwargs : dict or list of dicts, optional, defaults to None
            Additional arguments for plotters of components in `src`.
            If `dict`, defines a mapping from a component or a tuple of them to `plot` arguments, which are stored as
            `dict`s.
            If `list`, each element is a `dict` with arguments for the corresponding component in `src`.
        max_width : float, optional, defaults to 20
            Maximal figure width, measured in inches.
        title : str or dict, optional, defaults to "{src}: {index}"
            Title of subplots. If `dict`, should contain keyword arguments to pass to `matplotlib.axes.Axes.set_title`.
            In this case, the title string is stored under the `label` key.

            The title string may contain variables enclosed in curly braces that are formatted as python f-strings as
            follows:
            - "src" is substituted with the component name of the subplot,
            - "index" is substituted with the index of the current batch element,
            - All other variables are popped from the `title` `dict`.
        save_to : str or dict, optional, defaults to None
            If `str`, a path to save the figure to.
            If `dict`, should contain keyword arguments to pass to `matplotlib.pyplot.savefig`. In this case, the path
            is stored under the `fname` key.
            Otherwise, the figure is not saved.
        common_kwargs : misc, optional
            Additional common arguments to all plotters of components in `src`.

        Returns
        -------
        self : SeismicBatch
            The batch unchanged.

        Raises
        ------
        ValueError
            If the length of `src_kwargs` when passed as a list does not match the length of `src`.
            If any of the components' `plot` method is not decorated with `plotter` decorator.
        """
        # Construct a list of plot kwargs for each component in src
        src_list = to_list(src)
        if src_kwargs is None:
            src_kwargs = [{} for _ in range(len(src_list))]
        elif isinstance(src_kwargs, dict):
            src_kwargs = {src: src_kwargs[keys] for keys in src_kwargs for src in to_list(keys)}
            src_kwargs = [src_kwargs.get(src, {}) for src in src_list]
        else:
            src_kwargs = to_list(src_kwargs)
            if len(src_list) != len(src_kwargs):
                raise ValueError("The length of src_kwargs must match the length of src")

        # Construct a grid of plotters with shape (len(self), len(src_list)) for each of the subplots
        plotters = [[] for _ in range(len(self))]
        for src, kwargs in zip(src_list, src_kwargs):  # pylint: disable=redefined-argument-from-local
            # Merge src kwargs with common kwargs and defaults
            plotter_params = getattr(getattr(self, src)[0].plot, "method_params", {}).get("plotter")
            if plotter_params is None:
                raise ValueError("plot method of each component in src must be decorated with plotter")
            kwargs = {"figsize": plotter_params["figsize"], "title": title, **common_kwargs, **kwargs}

            # Scale subplot figsize if its width is greater than max_width
            width, height = kwargs.pop("figsize")
            if width > max_width:
                height = height * max_width / width
                width = max_width

            title_template = kwargs.pop("title")
            args_to_unpack = set(to_list(plotter_params["args_to_unpack"]))

            for i, index in enumerate(self.flat_indices):
                # Unpack required plotter arguments by getting the value of specified component with given index
                unpacked_args = {}
                for arg_name in args_to_unpack & kwargs.keys():
                    arg_val = kwargs[arg_name]
                    if isinstance(arg_val, dict) and arg_name in arg_val:
                        arg_val[arg_name] = self._unpack_args(arg_val[arg_name], self[i])
                    else:
                        arg_val = self._unpack_args(arg_val, self[i])
                    unpacked_args[arg_name] = arg_val

                # Format subplot title
                if title_template is not None:
                    src_title = as_dict(title_template, key='label')
                    label = src_title.pop("label")
                    format_names = {name for _, name, _, _ in Formatter().parse(label) if name is not None}
                    format_kwargs = {name: src_title.pop(name) for name in format_names if name in src_title}
                    src_title["label"] = label.format(src=src, index=index, **format_kwargs)
                    kwargs["title"] = src_title

                # Create subplotter config
                subplot_config = {
                    "plotter": partial(getattr(self, src)[i].plot, **{**kwargs, **unpacked_args}),
                    "height": height,
                    "width": width,
                }
                plotters[i].append(subplot_config)

        # Flatten all the subplots into a row if a single component was specified
        if len(src_list) == 1:
            plotters = [sum(plotters, [])]

        # Wrap lines of subplots wider than max_width
        split_pos = []
        curr_width = 0
        for i, plotter in enumerate(plotters[0]):
            curr_width += plotter["width"]
            if curr_width > max_width:
                split_pos.append(i)
                curr_width = plotter["width"]
        plotters = sum([np.split(plotters_row, split_pos) for plotters_row in plotters], [])

        # Define axes layout and perform plotting
        fig_width = max(sum(plotter["width"] for plotter in plotters_row) for plotters_row in plotters)
        row_heights = [max(plotter["height"] for plotter in plotters_row) for plotters_row in plotters]
        fig = plt.figure(figsize=(fig_width, sum(row_heights)), tight_layout=True)
        gridspecs = fig.add_gridspec(len(plotters), 1, height_ratios=row_heights)

        for gridspecs_row, plotters_row in zip(gridspecs, plotters):
            n_cols = len(plotters_row)
            col_widths = [plotter["width"] for plotter in plotters_row]

            # Create a dummy axis if row width is less than fig_width in order to avoid row stretching
            if fig_width > sum(col_widths):
                col_widths.append(fig_width - sum(col_widths))
                n_cols += 1

            # Create a gridspec for the current row
            gridspecs_col = gridspecs_row.subgridspec(1, n_cols, width_ratios=col_widths)
            for gridspec, plotter in zip(gridspecs_col, plotters_row):
                plotter["plotter"](ax=fig.add_subplot(gridspec))

        if save_to is not None:
            save_kwargs = as_dict(save_to, key="fname")
            save_figure(fig, **save_kwargs)
        return self
