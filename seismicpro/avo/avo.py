"""Implements AmplitudeOffsetDistribution class."""

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import save_figure


class AmplitudeOffsetDistribution:
    """A class representing an amplitude versus offset distribution.

    The Amplitude versus Offset (AVO) distribution showcases the variation of amplitude along the offset.

    The distribution is calculated as follows:
    1. Group traces into gathers based on the specified header column(s).
    2. Divide each gather into bins of the specified size and calculate the mean amplitude statistics value for each
       bin in each gather.
    Additionally, the mean of amplitude statistics and it's polynomial approximation for each bin are computed.

    The AmplitudeOffsetDistribution instance can be created either from survey using `from_survey` method or from
    headers using `__init__`. In both cases, headers should contain a column with precalculated amplitude statistics
    for each trace. The simplest way to calculate it is to call :func:`~Gather.calculate_amplitude_statistics` for all
    gathers in the survey.

    Two metrics are also collected:
    - `metric_std`: The mean standard deviation of amplitude statistics across all bins.
    - `metric_corr`: The correlation between the mean amplitude statistics per bin and its approximation by a
                     `pol_degree` degree polynomial.

    Parameters
    ----------
    headers : pd.DataFrame
        Trace headers of survey to calculate AVO distribution for.
    avo_column : str
        Headers column with computed amplitude statistics.
    bin_size : int or array-like
        If `int`, all bins will have the same size. Otherwise, an array-like with sizes for each bin.
        Measured in meters.
    indexed_by : str
        Headers column(s) used to group traces into gathers.
    name : str or None, optional, defaults to "Amplitude vs Offset Distribution"
        An AVO distribution instance name.
    pol_degree : int, optional, defaults to 3
        The degree of the polynomial used for approximation.

    Attributes
    ----------
    avo_column : str
        Header column with computed amplitude statistics.
    name : str
        The name of the AVO distribution instance to be used in plot methods.
    stats_df : pd.DataFrame
        A DataFrame containing statistics for each bin within each gather.
    bins_df : pd.DataFrame
        A DataFrame containing mean amplitude values per bin and it's polynomial approximation.
    metric_std : np.float64
        The mean standard deviation of amplitude statistics across all bins.
    metric_corr : np.float64
        The correlation between the mean amplitude statistics per bin and its approximation by a `pol_degree` degree
        polynomial.
    """
    def __init__(self, headers, avo_column, bin_size, indexed_by, name=None, pol_degree=3):
        if "offset" not in headers:
            raise ValueError("Missing offset header")
        self.avo_column = avo_column
        self.name = "Amplitude vs Offset Distribution" if name is None else name

        headers = headers.copy(deep=False)
        headers.reset_index(inplace=True)
        headers["offset"] = headers["offset"].abs()  # Avoid negative offsets

        headers = pl.from_pandas(headers, rechunk=False)
        if isinstance(bin_size, (int, np.integer)):
            bin_bounds = np.arange(0, headers["offset"].max()+bin_size, bin_size)
        else:
            bin_bounds = np.cumsum([0, *bin_size])

        # Find bin index for each trace in headers and subtract index by 1 to start at offset 0 instead of `bin_size`.
        headers = headers.with_columns(
            (pl.lit(bin_bounds)
               .search_sorted(pl.col("offset"), side="right") - 1)
               .clip(0, len(bin_bounds))
               .alias("bin_ix")
        )

        # Replace bin indices to the actual offset values
        headers = headers.with_columns(pl.lit(bin_bounds).take(pl.col("bin_ix")).alias("bin"))

        # Find for each gather mean amplitude statistics value in each bin
        stats_df = headers.groupby([*indexed_by, "bin"]).agg(pl.col(avo_column).mean())

        bins_groupby = stats_df.groupby("bin")
        # Compute mean amplitude statistics value for each bin
        bins_df = bins_groupby.agg(pl.col(avo_column).mean())

        # Add polynomial approximation for every bin
        bins_df = self._calculate_bin_polynomial(bins_df, pol_degree=pol_degree)

        # Metrics
        self.metric_std = bins_groupby.agg(pl.col(self.avo_column).std(ddof=0))[self.avo_column].mean()
        self.metric_corr = bins_df.select(pl.corr(self.avo_column, "bins_approx")).item()

        self.stats_df = stats_df[["bin", self.avo_column]].to_pandas()
        # Sort `bins_df` before converting to avoid sorting in `plot`` method
        self.bins_df = bins_df.sort("bin").to_pandas()

    def _calculate_bin_polynomial(self, bins_df, pol_degree=3):
        """Calculate polynomial approximation of the mean amplitude statistics per bin."""
        not_nan_df = bins_df.filter(pl.col(self.avo_column).is_not_null())

        poly = np.polyfit(not_nan_df["bin"], not_nan_df[self.avo_column], deg=pol_degree)
        bins_df = bins_df.with_columns(
            pl.when(pl.col(self.avo_column).is_null())
              .then(None)
              .otherwise(pl.col("bin").apply(lambda bins: np.polyval(poly, bins)))
              .alias("bins_approx")
        )
        return bins_df

    @classmethod
    def from_survey(cls, survey, avo_column, bin_size, indexed_by=None, name=None, pol_degree=3):
        """Compute amplitude versus offset distribution from the survey.

        Parameters
        ----------
        survey : Survey
            The survey from which to compute the distribution.
         avo_column : str
            Headers column with computed amplitude statistics.
        bin_size : int or array-like
            If `int`, all bins will have the same size. Otherwise, an array-like with sizes for each bin.
            Measured in meters.
        indexed_by : str
            Headers column(s) used to group traces into gathers.
        name : str or None, optional, defaults to "Amplitude vs Offset Distribution"
            A distribution name that will be used for visualization.
        pol_degree : int, optional, defaults to 3
            The degree of the polynomial used for approximation.

        Returns
        -------
        self : AmplitudeOffsetDistribution
            Created amplitude versus offset distribution instance.
        """
        indexed_by = indexed_by if indexed_by is not None else survey.indexed_by
        name = name if name is not None else survey.name
        return cls(headers=survey.headers, avo_column=avo_column, bin_size=bin_size, indexed_by=indexed_by, name=name,
                   pol_degree=pol_degree)

    def plot(self, show_qc=False, show_poly=False, title=None, figsize=(15, 7), dot_size=3, avg_size=50, dpi=100,
             save_to=None):
        """Plot the amplitude versus offset distribution.

        This plot includes scatter points for individual gather' amplitude statistics per bin and markers representing
        the mean amplitude statistics within each bin. It can also optionally show a polynomial approximation curve and
        quality control metrics.

        Parameters
        ----------
        show_qc : bool, optional, defaults to False
            Whether to display quality control metrics in the title.
        show_poly : bool, optional, defaults to False
            Whether to show the polynomial approximation curve.
        title : str, optional
            Title for the plot. If None, `self.name` is used.
        figsize : tuple, optional, defaults to (15, 7)
            Figure size. Measured in inches.
        dot_size : int, optional, defaults to 3
            Size of the markers for individual gather amplitude statistics.
        avg_size : int, optional, defaults to 50
            Size of markers for mean amplitude statistics values in bins.
        dpi : int, optional, defaults to 100
            Resolution of the saved figure. Measured in inches.
        save_to : str, optional
            If provided, the path to save the figure. If None, the figure is not saved.
        """
        # TODO: How to name plots?
        fig, ax = plt.subplots(figsize=figsize)
        self.stats_df.plot(x="bin", y=self.avo_column, kind="scatter", ax=ax, s=dot_size, label="Gather stats in bin")
        self.bins_df.plot(x="bin", y=self.avo_column, kind="scatter", ax=ax, s=avg_size, marker='v', color='r',
                          grid=True, label="Mean stats in bin")

        if show_poly:
            ax.plot(self.bins_df["bin"], self.bins_df["bins_approx"], "--", c='g', zorder=3,
                    label="Mean stats approximation")
        title = self.name if title is None else title
        title += f"\nstd: {self.metric_std:.4}\ncorr: {self.metric_corr:.4}" if show_qc else ""
        self._finalize_plot(fig, ax, title, save_to, dpi)

    def plot_std(self, *args, align=False, title=None, figsize=(15, 7), dpi=100, save_to=None):
        """Plot the standard deviation of amplitude versus offset distribution(s). The plot is allowed to compare
        multiple distribution on the same figure by passing them one after another.

        Parameters
        ----------
        *args : AmplitudeOffsetDistribution
            Additional amplitude versus offset distributions to draw.
        align : bool, optional, defaults to False
            Whether to align the means of the distribution to a common mean value.
        title : str, optional
            The title of the plot.
        figsize : tuple, optional, defaults to (15, 7)
            Figure size. Measured in inches.
        dpi : int, optional, defaults to 100
            Resolution of the saved figure. Measured in inches.
        save_to : str, optional
            If provided, the path to save the figure. If None, the figure is not saved.
        """
        avos = [self, *args]

        if align:
            global_mean = np.nanmean([avo.stats_df[avo.avo_column] for avo in avos])
        with plt.style.context("seaborn-v0_8-darkgrid"):
            fig, ax = plt.subplots(figsize=figsize)
            for avo in avos:
                stats = avo.stats_df
                if align:
                    stats = stats.copy(deep=False)
                    stats[avo.avo_column] = stats[avo.avo_column] + global_mean - np.nanmean(stats[avo.avo_column])
                sns.lineplot(data=stats, x="bin", y=avo.avo_column, errorbar="sd", ax=ax, label=avo.name)
            self._finalize_plot(fig, ax, title, save_to, dpi)

    def _finalize_plot(self, fig, ax, title, save_to, dpi):
        """Add axes labels, title, legend and save figure if needed."""
        ax.set_xlabel("Offset")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.legend()
        if save_to is not None:
            save_figure(fig, save_to, dpi=dpi)
