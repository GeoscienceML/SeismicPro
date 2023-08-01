"""Implements a metric that tracks a pipeline in which it was calculated and allows for automatic plotting of batch
components on its interactive maps"""

from inspect import signature
from functools import partial

import numpy as np
from batchflow import Pipeline

from .metric import Metric
from ..utils import to_list


class PipelineMetric(Metric):
    """Define a metric that tracks a pipeline in which it was calculated and allows for automatic plotting of batch
    components on its interactive maps.

    Examples
    --------
    Define a metric, that calculates standard deviation of gather amplitudes:
    >>> class StdMetric(PipelineMetric):
    ...     name = "std"
    ...     is_lower_better = None
    ...     min_value = 0
    ...     max_value = None
    ...     args_to_unpack = "gather"
    ...
    ...     def __call__(self, gather):
    ...         return gather.data.std()
    Note that the defined `__call__` method operates on each batch item independently.

    Calculate the metric for a given dataset:
    >>> survey = Survey(path, header_index="FieldRecord", header_cols=["SourceY", "SourceX", "offset"], name="raw")
    >>> dataset = SeismicDataset(survey)
    >>> pipeline = (dataset
    ...     .pipeline()
    ...     .load(src="raw")
    ...     .calculate_metric(StdMetric, gather="raw", save_to=V("map", mode="a"))
    ... )
    >>> pipeline.run(batch_size=16, n_epochs=1)

    `PipelineMetric` tracks a pipeline in which it was calculated. This allows reconstructing the batch used to compute
    the metric and plot its components on click on the interactive metric map:
    >>> std_map = pipeline.v("map")
    >>> std_map.plot(interactive=True, plot_component="raw")

    If a `pipeline` argument is passed, it will be used instead of the one used to calculate the metric:
    >>> plot_pipeline = Pipeline().load(src="raw").sort(src="raw", dst="sorted", by="offset")
    >>> std_map.plot(interactive=True, pipeline=plot_pipeline, plot_component="sorted")

    If several components are passed, a multiview plot is created:
    >>> std_map.plot(interactive=True, pipeline=plot_pipeline, plot_component=["raw", "sorted"])

    A `PipelineMetric` allows for defining default views. Each of them must accept the same arguments used by
    `__call__` along with the axes to plot on which will be passed as an `ax` keyword argument. Each of the views must
    be listed in the `views` class attribute. The following class extends `StdMetric` with two views: one plotting the
    gather used to calculate the metric itself and the other plotting the same gather sorted by offset.
    >>> class PlotStdMetric(StdMetric):
    ...     views = ("plot", "plot_sorted")
    ...
    ...     def plot(self, gather, ax, **kwargs):
    ...         return gather.plot(ax=ax, **kwargs)
    ...
    ...     def plot_sorted(self, gather, ax, **kwargs):
    ...         return gather.sort(by="offset").plot(ax=ax, **kwargs)

    In this case an interactive plot of the map may be constructed without any extra arguments:
    >>> pipeline = (dataset
    ...     .pipeline()
    ...     .load(src="raw")
    ...     .calculate_metric(PlotStdMetric, gather="raw", save_to=V("map", mode="a"))
    ... )
    >>> pipeline.run(batch_size=16, n_epochs=1)
    >>> std_map = pipeline.v("map")
    >>> std_map.plot(interactive=True)

    Parameters
    ----------
    name : str, optional
        Metric name, overrides default name if given.

    Attributes
    ----------
    args_to_unpack : str or list of str or "all"
        Arguments to unpack before passing to `__call__` and views. Unpacking is performed in the following way:
        * If argument value is `str`, it is treated as a batch component name to get the actual argument from,
        * If argument value is an array-like whose length matches the length of the batch, its elements are passed to
          `__call__` method of the corresponding batch item.
        * Otherwise the argument value is passed to `__call__` methods for all batch items.
        If "all", tries to unpack all the arguments.
    views : str or iterable of str
        Default views of the metric to display on click on a metric map in interactive mode.
    dataset : SeismicDataset
        The dataset for which the metric was calculated.
    plot_pipeline : Pipeline
        The `pipeline` up to the `calculate_metric` method.
    calculate_metric_args : tuple
        Positional arguments passed to `calculate_metric` call. May contain non-evaluated named expressions.
    calculate_metric_kwargs : dict
        Keyword arguments passed to `calculate_metric` call. May contain non-evaluated named expressions.
    """
    args_to_unpack = None

    def __init__(self, name=None):
        super().__init__(name=name)

        # Attributes set after context binding
        self.dataset = None
        self.plot_pipeline = None
        self.calculate_metric_args = None
        self.calculate_metric_kwargs = None

    def __call__(self, value):
        """Return an already calculated metric. May be overridden in child classes."""
        return value

    def bind_context(self, metric_map, pipeline, calculate_metric_index):
        """Process metric evaluation context: memorize the dataset used, parameters passed to `calculate_metric` and
        a part of the execution pipeline up to the `calculate_metric` call."""
        _ = metric_map
        self.dataset = pipeline.dataset

        # Slice the pipeline in which the metric was calculated up to its calculate_metric call
        # pylint: disable=protected-access
        calculate_metric_indices = [i for i, action in enumerate(pipeline._actions)
                                      if action["name"] == "calculate_metric"]
        calculate_metric_action_index = calculate_metric_indices[calculate_metric_index]
        actions = pipeline._actions[:calculate_metric_action_index]
        self.plot_pipeline = Pipeline(pipeline=pipeline, actions=actions)

        # Get args and kwargs of the calculate_metric call with possible named expressions in them
        self.calculate_metric_args = pipeline._actions[calculate_metric_action_index]["args"]
        self.calculate_metric_kwargs = pipeline._actions[calculate_metric_action_index]["kwargs"]
        # pylint: enable=protected-access

    def get_calc_signature(self):
        """Get a signature of the metric calculation function."""
        return signature(self.__call__)

    def unpack_calc_args(self, batch, *args, **kwargs):
        """Unpack arguments for metric calculation depending on the `args_to_unpack` class attribute and return them
        with the first unpacked `calc` argument. If `args_to_unpack` equals "all", tries to unpack all the passed
        arguments.

        Unpacking is performed in the following way:
        * If argument value is `str`, it is treated as a batch component name to get the actual argument from,
        * If argument value is an array-like whose length matches the length of the batch, its elements are passed to
          `calc` methods for the corresponding batch items.
        * Otherwise the argument value is passed to `calc` methods for all batch items.
        """
        sign = self.get_calc_signature()
        bound_args = sign.bind(*args, **kwargs)

        # Determine arguments to unpack
        if self.args_to_unpack is None:
            args_to_unpack = set()
        elif self.args_to_unpack == "all":
            args_to_unpack = {name for name, param in sign.parameters.items()
                                   if param.kind not in {param.VAR_POSITIONAL, param.VAR_KEYWORD}}
        else:
            args_to_unpack = set(to_list(self.args_to_unpack))

        # Convert the value of each argument to an array-like matching the length of the batch
        packed_args = {}
        for arg, val in bound_args.arguments.items():
            if arg in args_to_unpack:
                if isinstance(val, str):
                    packed_args[arg] = getattr(batch, val)
                elif isinstance(val, (tuple, list, np.ndarray)) and len(val) == len(batch):
                    packed_args[arg] = val
                else:
                    packed_args[arg] = [val] * len(batch)
            else:
                packed_args[arg] = [val] * len(batch)

        # Extract the values of the first calc argument to use them as a default source for coordinates calculation
        first_arg = packed_args[list(sign.parameters.keys())[0]]

        # Convert packed args dict to a list of calc args and kwargs for each of the batch items
        unpacked_args = []
        for values in zip(*packed_args.values()):
            bound_args.arguments = dict(zip(packed_args.keys(), values))
            unpacked_args.append((bound_args.args, bound_args.kwargs))
        return unpacked_args, first_arg

    def eval_calc_args(self, batch):
        """Evaluate named expressions in arguments passed to the `__call__` method and unpack arguments for the first
        batch item."""
        sign = signature(batch.calculate_metric)
        bound_args = sign.bind(*self.calculate_metric_args, **self.calculate_metric_kwargs)
        bound_args.apply_defaults()
        # pylint: disable=protected-access
        calc_args = self.plot_pipeline._eval_expr(bound_args.arguments["args"], batch=batch)
        calc_kwargs = self.plot_pipeline._eval_expr(bound_args.arguments["kwargs"], batch=batch)
        # pylint: enable=protected-access
        args, _ = self.unpack_calc_args(batch, *calc_args, **calc_kwargs)
        return args[0]

    def make_batch(self, index, pipeline=None):
        """Construct a batch for given `index` and execute the `pipeline` for it. If `pipeline` is not given,
        `plot_pipeline` is used."""
        subset_index = [[] for _ in range(self.dataset.n_parts)]
        subset_index[index[0]] = [index[1:] if len(index) > 2 else index[1]]
        batch = self.dataset.create_subset(subset_index).next_batch(1, shuffle=False)
        if pipeline is None:
            pipeline = self.plot_pipeline
        return pipeline.execute_for(batch)

    def plot_component(self, ax, coords, index, plot_component, pipeline, **kwargs):
        """Construct a batch by its index and plot the given component."""
        _ = coords
        batch = self.make_batch(index, pipeline)
        item = getattr(batch, plot_component)[0]
        item.plot(ax=ax, **kwargs)

    def plot_view(self, ax, coords, index, view_fn, **kwargs):
        """Plot a given metric view."""
        _ = coords
        batch = self.make_batch(index)
        calc_args, calc_kwargs = self.eval_calc_args(batch)
        return view_fn(*calc_args, ax=ax, **calc_kwargs, **kwargs)

    def get_views(self, plot_component=None, pipeline=None, **kwargs):
        """Get metric views by parameters passed to interactive metric map plotter. If `plot_component` is given,
        batch components are displayed. Otherwise defined metric views are shown."""
        if plot_component is not None:
            views = [partial(self.plot_component, plot_component=component, pipeline=pipeline)
                     for component in to_list(plot_component)]
            return views, kwargs

        view_fns = [getattr(self, view) for view in to_list(self.views)]
        return [partial(self.plot_view, view_fn=view_fn) for view_fn in view_fns], kwargs


class FunctionalMetric(PipelineMetric):
    """Construct a metric from a callable by executing it in `__call__` with all passed arguments."""
    args_to_unpack = "all"

    def __init__(self, func, name=None):
        if not callable(func):
            raise ValueError("func must be callable")
        self.func = func
        super().__init__(name=name)

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(func={self.func.__name__}, name='{self.name}')"

    def __call__(self, *args, **kwargs):
        """Calculate `func` with given parameters."""
        return self.func(*args, **kwargs)

    def get_calc_signature(self):
        """Get a signature of `func`."""
        return signature(self.func)


def define_pipeline_metric(metric, metric_name=None):
    """Return an instance of `PipelineMetric` from `metric`, which may be a callable or an instance or a subclass of
    `PipelineMetric`. If given, `metric_name` defines a name of the constructed metric."""
    if isinstance(metric, PipelineMetric):  # Instantiated metric
        return metric.set_name(metric_name)
    if isinstance(metric, type) and issubclass(metric, PipelineMetric):  # Non-instantiated metric
        return metric(name=metric_name)
    if callable(metric):
        return FunctionalMetric(func=metric, name=metric_name)
    raise TypeError("metric must be either a callable or an instance or a subclass of PipelineMetric")
