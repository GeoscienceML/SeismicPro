"""AVO"""

import numpy as np
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
        if isinstance(bin_size, (int, np.integer)):
            bin_bounds = np.arange(0, headers["offset"].max()+bin_size, bin_size)
        else:
            bin_bounds = np.cumsum([0, *bin_size])

        # Subtract 1 to start at offset 0 instead of `bin_size`
        bin_bounds_ixs = np.searchsorted(bin_bounds, headers["offset"], side='right') - 1
        headers["bin"] = bin_bounds[np.clip(bin_bounds_ixs, 0, len(bin_bounds))]
        self.stats_df = headers.groupby([*to_list(indexed_by), "bin"], as_index=False)[avo_column].mean()
        self.bins_df = self.stats_df.groupby("bin", as_index=False)[avo_column].mean()

        # Metrics
        self.metrics = {}
        self.qc(pol_degree=pol_degree)

    @classmethod
    def from_survey(cls, survey, avo_column, bin_size, indexed_by=None, name=None, pol_degree=3):
        indexed_by = indexed_by if indexed_by is not None else survey.indexed_by
        name = name if name is not None else survey.name
        return cls(headers=survey.headers, avo_column=avo_column, bin_size=bin_size, indexed_by=indexed_by, name=name,
                   pol_degree=pol_degree)

    def qc(self, pol_degree=3):
        if "std" not in self.metrics:
            self.metrics["std"] = self.stats_df.groupby("bin")[self.avo_column].apply(np.nanstd).mean()
        self.metrics["corr"] = self._calculate_correlation_with_polynomial(pol_degree=pol_degree)

    def _calculate_correlation_with_polynomial(self, pol_degree):
        mask = ~self.bins_df[self.avo_column].isna()
        not_nan_bins = self.bins_df[mask]
        poly = np.polyfit(not_nan_bins["bin"], not_nan_bins[self.avo_column], deg=pol_degree)

        bins_approx = np.full(len(mask), np.nan)
        bins_approx[mask] = np.polyval(poly, not_nan_bins["bin"])
        self.bins_df["bins_approx"] = bins_approx
        return np.corrcoef(not_nan_bins[self.avo_column], bins_approx[mask])[0][1]

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
            title = "" if title is None else title
            for name, value in self.metrics.items():
                sep = "\n" if title else ""
                title = title + sep + f"{name} : {value:.4}"
        ax.set_xlabel('Offset')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.legend()
        if save_to is not None:
            save_figure(fig, save_to, dpi=dpi)
