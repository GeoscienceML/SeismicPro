import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from .metrics import TravelTimeMetric, TRAVELTIME_QC_METRICS
from .utils import get_uphole_correction_method
from ..metrics import initialize_metrics
from ..utils import to_list, align_args, ForPoolExecutor
from ..const import HDR_FIRST_BREAK


class TravelTimeDataset:
    def __init__(self, survey, grid, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto"):
        self.grid = grid

        aligned_args = align_args(survey, first_breaks_header, uphole_correction_method)
        self.survey_list = aligned_args[0]
        self.first_breaks_header_list = aligned_args[1]
        self.uphole_correction_method_list = [get_uphole_correction_method(survey, method)
                                              for survey, method in zip(aligned_args[0], aligned_args[2])]

        # Extract information about source and receiver location and the corresponding traveltime from given surveys
        survey_list_data = self._get_survey_list_data(self.survey_list, self.first_breaks_header_list,
                                                      self.uphole_correction_method_list)
        self.source_coords, self.source_elevations = survey_list_data[:2]
        self.receiver_coords, self.receiver_elevations = survey_list_data[2:4]
        self.true_traveltimes, self.target_traveltimes, self.traveltime_corrections = survey_list_data[4:]
        self.pred_traveltimes = None

    @property
    def has_predictions(self):
        return self.pred_traveltimes is not None

    @staticmethod
    def _get_survey_data(survey, first_breaks_header, uphole_correction_method):
        source_coords = np.require(survey[["SourceX", "SourceY"]], dtype=np.float32)
        source_elevations = np.require(survey["SourceSurfaceElevation"], dtype=np.float32)

        receiver_coords = np.require(survey[["GroupX", "GroupY"]], dtype=np.float32)
        receiver_elevations = np.require(survey["ReceiverGroupElevation"], dtype=np.float32)

        true_traveltimes = np.require(survey[first_breaks_header], dtype=np.float32)
        target_traveltimes = true_traveltimes

        if uphole_correction_method == "time":
            traveltime_corrections = np.require(survey["SourceUpholeTime"], dtype=np.float32)
            target_traveltimes = target_traveltimes + traveltime_corrections
        elif uphole_correction_method == "depth":
            traveltime_corrections = np.zeros_like(target_traveltimes)
            source_elevations = source_elevations - np.require(survey["SourceDepth"], dtype=np.float32)
        else:
            traveltime_corrections = np.zeros_like(target_traveltimes)

        survey_data = (source_coords, source_elevations, receiver_coords, receiver_elevations,
                       true_traveltimes, target_traveltimes, traveltime_corrections)
        return survey_data

    @classmethod
    def _get_survey_list_data(cls, survey_list, first_breaks_header_list, uphole_correction_method_list):
        survey_iterator = zip(survey_list, first_breaks_header_list, uphole_correction_method_list)
        survey_list_data = list(zip(*[cls._get_survey_data(*params) for params in survey_iterator]))
        survey_list_data = [seq[0] if len(survey_list) == 1 else np.concatenate(seq) for seq in survey_list_data]
        return survey_list_data

    def store_predictions_to_survey(self, predicted_first_breaks_header="PredictedFirstBreak"):
        if not self.has_predictions:
            raise ValueError

        split_indices = np.cumsum([survey.n_traces for survey in self.survey_list[:-1]])
        pred_traveltimes = np.split(self.pred_traveltimes, split_indices)
        data_iterator = zip(align_args(self.survey_list, pred_traveltimes, predicted_first_breaks_header))
        for survey, traveltimes, header in data_iterator:
            survey[header] = traveltimes

    # Evaluation of predictions

    def evaluate(self):
        if not self.has_predictions:
            raise ValueError
        return np.abs(self.pred_traveltimes - self.true_traveltimes).mean()

    # Near-surface model QC

    @staticmethod
    def _calc_metrics(metrics, gather_data_list):
        return [[metric(*gather_data) for metric in metrics] for gather_data in gather_data_list]

    def qc(self, metrics=None, by="source", id_cols=None, chunk_size=10, n_workers=None, bar=True):
        if not self.has_predictions:
            raise ValueError

        if metrics is None:
            metrics = TRAVELTIME_QC_METRICS
        metrics, is_single_metric = initialize_metrics(metrics, metric_class=TravelTimeMetric)

        by_to_cols = {
            "source": ("source_id_cols", ["SourceX", "SourceY"]),
            "shot": ("source_id_cols", ["SourceX", "SourceY"]),
            "receiver": ("receiver_id_cols", ["GroupX", "GroupY"]),
            "rec": ("receiver_id_cols", ["GroupX", "GroupY"]),
        }
        id_cols_attr, coords_cols = by_to_cols.get(by.lower())
        if id_cols_attr is None:
            raise ValueError(f"by must be one of {', '.join(by_to_cols.keys())} but {by} given.")

        if id_cols is None:
            id_cols_list = [getattr(survey, id_cols_attr) for survey in self.survey_list]
            if any(item != id_cols_list[0] for item in id_cols_list):
                raise ValueError("source/receiver id columns must be the same for all surveys")
            id_cols = id_cols_list[0]
            if id_cols is None:
                raise ValueError
        id_cols = to_list(id_cols)

        id_cols_df_list = [survey.get_headers(id_cols) for survey in self.survey_list]
        if len(self.survey_list) == 1:
            qc_df = id_cols_df_list[0]
        else:
            id_cols = ["Part"] + id_cols
            for i, df in enumerate(id_cols_df_list):
                df["Part"] = i
            qc_df = pd.concat(id_cols_df_list, ignore_index=True)
        qc_df[["SourceX", "SourceY"]] = self.source_coords
        qc_df[["GroupX", "GroupY"]] = self.receiver_coords
        qc_df["True"] = self.true_traveltimes
        qc_df["Pred"] = self.pred_traveltimes

        qc_df = pl.from_pandas(qc_df, rechunk=False, include_index=False)
        gather_data_dict = qc_df.partition_by(id_cols, maintain_order=True, as_dict=True)
        gather_data_list = [(df.select("SourceX", "SourceY").to_numpy(), df.select("GroupX", "GroupY").to_numpy(),
                             df.get_column("True").to_numpy(), df.get_column("Pred").to_numpy())
                            for df in gather_data_dict.values()]
        coords = pd.DataFrame(np.stack([df[coords_cols].row(0) for df in gather_data_dict.values()]),
                              columns=coords_cols)
        index = pd.DataFrame(np.stack(list(gather_data_dict.keys())), columns=id_cols)

        n_gathers = len(gather_data_list)
        n_chunks, mod = divmod(n_gathers, chunk_size)
        if mod:
            n_chunks += 1
        if n_workers is None:
            n_workers = os.cpu_count()
        n_workers = min(n_chunks, n_workers)
        executor_class = ForPoolExecutor if n_workers == 1 else ProcessPoolExecutor

        futures = []
        with tqdm(total=n_gathers, desc="Gathers processed", disable=not bar) as pbar:
            with executor_class(max_workers=n_workers) as pool:
                for i in range(n_chunks):
                    gather_data_chunk = gather_data_list[i * chunk_size : (i + 1) * chunk_size]
                    future = pool.submit(self._calc_metrics, metrics, gather_data_chunk)
                    future.add_done_callback(lambda fut: pbar.update(len(fut.result())))
                    futures.append(future)

        results = sum([future.result() for future in futures], [])
        context = {"near_surface_model": self, "survey_list": self.survey_list,
                   "first_breaks_header_list": self.first_breaks_header_list, "gather_data_dict": gather_data_dict}
        metrics_maps = [metric.provide_context(**context).construct_map(coords, values, index=index)
                        for metric, values in zip(metrics, zip(*results))]
        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps
