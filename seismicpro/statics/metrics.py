from functools import partial

import numpy as np
from scipy.optimize import minimize

from ..metrics import Metric


class TravelTimeMetric(Metric):
    def __init__(self, name=None):
        super().__init__(name=name)

        # Attributes set after context binding
        self.near_surface_model = None
        self.survey_list = None
        self.first_breaks_header_list = None
        self.gather_data_dict = None

    def bind_context(self, metric_map, near_surface_model, survey_list, first_breaks_header_list, gather_data_dict):
        self.near_surface_model = near_surface_model
        index_cols = metric_map.index_cols if len(survey_list) == 1 else metric_map.index_cols[1:]
        self.survey_list = [survey.reindex(index_cols) for survey in survey_list]
        self.first_breaks_header_list = first_breaks_header_list
        self.gather_data_dict = gather_data_dict

    def get_gather(self, index, sort_by=None):
        pred_traveltimes = self.gather_data_dict[index].get_column("Pred").to_numpy()
        if len(self.survey_list) == 1:
            part = 0
        else:
            part = index[0]
            index = index[1:]
        survey = self.survey_list[part]
        gather = survey.get_gather(index, copy_headers=True)
        true_first_breaks_header = self.first_breaks_header_list[part]
        pred_first_breaks_header = "Predicted " + true_first_breaks_header
        gather[pred_first_breaks_header] = pred_traveltimes
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        return gather, true_first_breaks_header, pred_first_breaks_header

    def plot_on_click(self, ax, coords, index, sort_by=None, **kwargs):
        _ = coords
        gather, true_first_breaks_header, pred_first_breaks_header = self.get_gather(index, sort_by)
        gather.plot(ax=ax, event_headers=[true_first_breaks_header, pred_first_breaks_header], **kwargs)

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot_on_click, sort_by=sort_by)], kwargs


class MeanAbsoluteTravelTimeError(TravelTimeMetric):
    min_value = 0
    is_lower_better = True

    def __call__(self, source_coords, receiver_coords, true_traveltimes, pred_traveltimes):
        _ = source_coords, receiver_coords
        return np.abs(true_traveltimes - pred_traveltimes).mean()


class GeometryError(TravelTimeMetric):
    min_value = 0
    is_lower_better = True

    def __init__(self, reg=0.01, name=None):
        self.reg = reg
        super().__init__(name=name)

    def __repr__(self):
        """String representation of the metric."""
        return f"{type(self).__name__}(reg={self.reg}, name='{self.name}')"

    @staticmethod
    def sin(x, amp, phase):
        return amp * np.sin(x + phase)

    @classmethod
    def loss(cls, params, x, y, reg):
        return np.abs(y - cls.sin(x, *params)).mean() + reg * params[0]**2

    @classmethod
    def fit(cls, azimuth, diff, reg):
        fit_result = minimize(cls.loss, x0=[0, 0], args=(azimuth, diff - diff.mean(), reg),
                              bounds=((None, None), (-np.pi, np.pi)), method="Nelder-Mead", tol=1e-5)
        return fit_result.x

    def __call__(self, source_coords, receiver_coords, true_traveltimes, pred_traveltimes):
        diff = true_traveltimes - pred_traveltimes
        x, y = (receiver_coords - source_coords).T
        azimuth = np.arctan2(y, x)
        params = self.fit(azimuth, diff, reg=self.reg)
        return abs(params[0])

    def plot_diff_by_azimuth(self, ax, coords, index, **kwargs):
        _ = coords, kwargs
        gather, true_first_breaks_header, pred_first_breaks_header = self.get_gather(index)
        diff = gather[true_first_breaks_header] - gather[pred_first_breaks_header]
        x, y = (gather[["GroupX", "GroupY"]] - gather[["SourceX", "SourceY"]]).T
        azimuth = np.arctan2(y, x)

        params = self.fit(azimuth, diff, reg=self.reg)
        ax.scatter(azimuth, diff)
        azimuth = np.linspace(-np.pi, np.pi, 100)
        ax.plot(azimuth, diff.mean() + self.sin(azimuth, *params))

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot_on_click, sort_by=sort_by), self.plot_diff_by_azimuth], kwargs


TRAVELTIME_QC_METRICS = [MeanAbsoluteTravelTimeError, GeometryError]
