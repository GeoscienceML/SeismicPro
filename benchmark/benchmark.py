"""Implements Benchmark class to choose optimal inbatch_parallel target for SeismicBatch methods"""

import dill
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from batchflow import Pipeline, CPUMonitor, C
from batchflow.research import Option, Research, EC


sns.set_theme(style="darkgrid")
class Benchmark:
    """A class aimed to find an optimal parallelization engine for methods decorated with
    :func:`~decorators.batch_method`.

    `Benchmark` runs experiments with all combinations of given parallelization engines (`targets`) and batch sizes for
    the specified method and measures the execution time. To get a more accurate time estimation, each experiment is
    repeated `n_iters` times.

    Simple usage of `Benchmark` contains three steps:
    1. Define Benchmark instance.
    2. Call `benchmark.run()` to run the benchmark.
    3. Call `benchmark.plot()` to plot the results.

    Parameters
    ----------
    method_name : str
        A name of the benchmarked method.
    targets : str or array of str
        Name(s) of target from :func:`~batchflow.batchflow.decorators.inbatch_parallel`.
    batch_sizes : int or array of int
        Batch size(s) to run the benchmark for.
    dataset : Dataset
        Dataset for which the benchmark is conducted.
    method_args : tuple, optional, defaults to None
        Additional positional arguments to the benchmarked method.
    method_kwargs : dict, optional, defaults to None
        Additional keyword arguments to the benchmarked method.
    root_pipeline : Pipeline, optional, defaults to None
        Pipeline that contains actions to be performed before the benchmarked method.

    Attributes
    ----------
    method_name : str
        A name of the benchmarked method.
    results : None or pd.DataFrame
        A DataFrame with benchmark results.
    domain : Domain
        Grid or parameters to provide the benchmark with.
    template_pipeline : Pipeline
        Pipeline that contains `root_pipeline`, benchmarked method, and dataset.
    """
    def __init__(self, method_name, targets, batch_sizes, dataset, method_args=None, method_kwargs=None,
                 root_pipeline=None):
        self.method_name = method_name
        self.results = None
        self.domain = Option('target', targets) * Option('batch_size', batch_sizes)

        method_args = () if method_args is None else method_args
        method_kwargs = {} if method_kwargs is None else method_kwargs
        # Add benchmarked method to the `root_pipeline` with `method_kwargs` and `target` from config.
        method_kwargs['target'] = C('target')
        root_pipeline = Pipeline() if root_pipeline is None else root_pipeline
        root_pipeline = getattr(root_pipeline, self.method_name)(*method_args, **method_kwargs)
        self.template_pipeline = root_pipeline << dataset

        # Run the pipeline once to precompile all numba callables
        self._warmup()

    def _warmup(self):
        """Run `self.template_pipeline` once."""
        (self.template_pipeline << {'target': 'for'}).next_batch(1)

    def save(self, path):
        """Pickle Benchmark to a file.

        Parameters
        ----------
        path : str
            A path to save the benchmark.

        Returns
        -------
        self : Benchmark
            Unchanged Benchmark.
        """
        with open(path, 'wb') as file:
            dill.dump(self, file)
        return self

    @staticmethod
    def load(path):
        """Unpickle Benchmark from a file.

        Parameters
        ----------
        path : str
            A path to a pickled benchmark.

        Returns
        -------
        benchmark : Benchmark
            Unpickled Benchmark.
        """
        with open(path, 'rb') as file:
            benchmark = dill.load(file)
        return benchmark

    def run(self, n_iters=10, shuffle=False, bar=True, env_meta=False):
        """Measure the execution time and CPU utilization of the benchmarked method for all combinations of targets and
        batch sizes.

        Parameters
        ----------
        n_iters : int, optional, defaults to 10
            The number of method executions to get a more accurate elapsed time estimation.
        shuffle : int or bool, defaults to False
            Specifies the randomization in the pipeline.
            If `False`: items go sequentially, one after another as they appear in the index;
            If `True`: items are shuffled randomly before each epoch;
            If int: a seed number for a random shuffle;
        bar : bool, optional, defaults to True
            Whether to use progress bar or not.
        env_meta : dict or bool, optional, defaults to False
            if dict, kwargs for :meth:`~batchflow.batchflow.research.attach_env_meta`
            if bool, whether to attach environment meta or not.

        Returns
        -------
        self : Benchmark
            Benchmark with computed results.
        """
        # Create research that will run pipeline with different parameters based on given `domain`
        research = (Research(domain=self.domain, n_reps=1)
            .add_callable(self._run_single_pipeline, config=EC(), n_iters=n_iters, shuffle=shuffle,
                          save_to=['Time', 'CPUMonitor'])
        ).run(n_iters=1, dump_results=False, parallel=False, workers=1, bar=bar, env_meta=env_meta)

        # Load benchmark's results.
        results = research.results.to_df().astype({"batch_size": np.int32}).set_index(['target', 'batch_size'])
        self.results = results[['Time', 'CPUMonitor']]
        return self

    def _run_single_pipeline(self, config, n_iters, shuffle):
        """Benchmark the method with a particular `batch_size` and `target`."""
        pipeline = self.template_pipeline << config
        with CPUMonitor() as cpu_monitor:
            pipeline.run(C('batch_size'), n_iters=n_iters, shuffle=shuffle, drop_last=True, notifier=False,
                         profile=True)

        # Processing the results for time costs.
        time_df = pipeline.show_profile_info(per_iter=True, detailed=False)
        action_name = f'{self.method_name} #{pipeline.num_actions-1}'
        run_time = time_df['total_time'].loc[:, action_name].values

        return run_time, cpu_monitor.data

    def plot(self, figsize=(12, 6), cpu_util=False):
        """Plot time and, optionally, CPU utilization versus `batch_size` for each `target`.

        The graph represents the average value and the standard deviation of the elapsed time or CPU
        utilization over `n_iters` iterations.

        Parameters
        ----------
        figsize : tuple, optional, defaults to (12, 6)
            Output plot size.
        cpu_util : bool, defaults to False
            If True, the CPU utilization is plotted next to the elapsed time plot.
        """
        results = self.results.drop(columns="CPUMonitor") if not cpu_util else self.results
        batch_sizes = np.unique(results.reset_index()["batch_size"])
        for col_name, col_series in results.items():
            sub_df = col_series.explode().reset_index()

            plt.figure(figsize=figsize)
            sns.lineplot(data=sub_df, x="batch_size", y=col_name, hue="target", errorbar='sd', marker='o')

            plt.title(f"{col_name} for {self.method_name} method")
            plt.xticks(ticks=batch_sizes, labels=batch_sizes)
            plt.ylim(0)
            plt.xlabel("Batch size")
            plt.ylabel("Time (s)" if col_name == "Time" else "CPU (%)")
        plt.show()
