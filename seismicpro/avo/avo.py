"""AVO"""

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import to_list, save_figure


class AmplitudeOffsetDistribution:
    def __init__(self, headers, avo_column, bin_size, indexed_by, name=None, pol_degree=3):
        if "offset" not in headers:
            raise ValueError("Missing offset header")
        self.avo_column = avo_column
        self.name = name

        headers = headers.copy(deep=False)
        headers.reset_index(inplace=True)
        headers['offset'] = headers['offset'].abs()

        headers = pl.from_pandas(headers, rechunk=False)
        # Avoid negative offsets
        if isinstance(bin_size, (int, np.integer)):
            bin_bounds = np.arange(0, headers["offset"].max()+bin_size, bin_size)
        else:
            bin_bounds = np.cumsum([0, *bin_size])

        # Subtract 1 to start at offset 0 instead of `bin_size`
        headers = headers.with_columns(
            (pl.lit(bin_bounds)
               .search_sorted(pl.col('offset'), side='right') - 1)
               .clip(0, len(bin_bounds))
               .alias('bin_ix')
        )

        # Change bin indices to actual offset values on the bounds
        headers = headers.with_columns(pl.lit(bin_bounds).take(pl.col('bin_ix')).alias('bin'))

        # Find for each gather mean AVO value in each bin
        stats_df = headers.groupby([*indexed_by, "bin"]).agg(pl.col(avo_column).mean())

        bins_groupby = stats_df.groupby("bin")
        # Compute mean AVO value for each bin
        bins_df = bins_groupby.agg(pl.col(avo_column).mean())

        # Add polynomial approximation for every bin
        bins_df = self._calculate_bin_polynomial(bins_df, pol_degree=pol_degree)

        # Metrics
        self.metric_std = bins_groupby.agg(pl.col(self.avo_column).std(ddof=0))[self.avo_column].mean()
        self.metric_corr = bins_df.select(pl.corr(self.avo_column, "bins_approx")).item()

        self.stats_df = stats_df[["bin", self.avo_column]].to_pandas()
        # Sort bins_df before converting to avoid sorting in `plot`` method
        self.bins_df = bins_df.sort("bin").to_pandas()

    def _calculate_bin_polynomial(self, bins_df, pol_degree=3):
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
        indexed_by = indexed_by if indexed_by is not None else survey.indexed_by
        name = name if name is not None else survey.name
        return cls(headers=survey.headers, avo_column=avo_column, bin_size=bin_size, indexed_by=indexed_by, name=name,
                   pol_degree=pol_degree)

    def plot(self, show_qc=False, show_poly=False, title=None, figsize=(15, 7), dot_size=3, avg_size=50, dpi=100,
             save_to=None):
        fig, ax = plt.subplots(figsize=figsize)
        self.stats_df.plot(x='bin', y=self.avo_column, kind='scatter', ax=ax, s=dot_size, label="Gather AVO in bin")
        self.bins_df.plot(x='bin', y=self.avo_column, kind='scatter', ax=ax, s=avg_size, marker='v', color='r',
                          grid=True, label="Mean AVO in bin")

        if show_poly:
            ax.plot(self.bins_df["bin"], self.bins_df["bins_approx"], '--', c='g', zorder=3,
                    label="Mean AVO approximation")
        self._finalize_plot(fig, ax, show_qc, title, save_to, dpi)

    def plot_std(self, *args, align=False, title=None, figsize=(15, 7), dpi=100, save_to=None):
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
                sns.lineplot(data=stats, x="bin", y=avo.avo_column, errorbar='sd', ax=ax, label=avo.name)
            self._finalize_plot(fig, ax, False, title, save_to, dpi)

    def _finalize_plot(self, fig, ax, show_qc, title, save_to, dpi):
        if show_qc:
            title = "" if title is None else title + "\n"
            title += f"std: {self.metric_std:.4}\ncorr: {self.metric_corr:.4}"
        ax.set_xlabel('Offset')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.legend()
        if save_to is not None:
            save_figure(fig, save_to, dpi=dpi)
