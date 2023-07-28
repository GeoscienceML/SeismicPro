import os
import math
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np
import pandas as pd
import polars as pl
from numba import njit, prange
from tqdm.auto import tqdm

from .dataloader import TensorDataLoader
from ..metrics import TravelTimeMetric, TRAVELTIME_QC_METRICS
from ..utils import get_uphole_correction_method
from ...metrics import initialize_metrics
from ...utils import to_list, align_args, ForPoolExecutor
from ...const import HDR_FIRST_BREAK


class TravelTimeDataset:
    def __init__(self, survey, grid, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto",
                 slowness_grid_size=500):
        self.grid = grid

        aligned_args = align_args(survey, first_breaks_header, uphole_correction_method)
        self.survey_list = aligned_args[0]
        self.first_breaks_header_list = aligned_args[1]
        self.uphole_correction_method_list = [get_uphole_correction_method(survey, method)
                                              for survey, method in zip(aligned_args[0], aligned_args[2])]

        # Extract information about source and receiver location and the corresponding traveltime from given surveys
        survey_list_data = self._get_survey_list_data(self.survey_list, self.first_breaks_header_list,
                                                      self.uphole_correction_method_list)
        source_coords, source_elevations, receiver_coords, receiver_elevations = survey_list_data[:4]
        true_traveltimes, target_traveltimes, traveltime_corrections = survey_list_data[4:]
        self.source_coords = torch.from_numpy(source_coords)
        self.source_elevations = torch.from_numpy(source_elevations)
        self.receiver_coords = torch.from_numpy(receiver_coords)
        self.receiver_elevations = torch.from_numpy(receiver_elevations)
        self.true_traveltimes = torch.from_numpy(true_traveltimes)
        self.pred_traveltimes = None
        self.target_traveltimes = torch.from_numpy(target_traveltimes)
        self.traveltime_corrections = torch.from_numpy(traveltime_corrections)

        # Get indices of grid coords and their weights for each source and receiver for further interpolation of
        # layered model parameters
        source_indices, source_weights = self._get_sensor_interpolation_params(source_coords, grid)
        self.source_indices = torch.from_numpy(source_indices)
        self.source_weights = torch.from_numpy(source_weights)
        receiver_indices, receiver_weights = self._get_sensor_interpolation_params(receiver_coords, grid)
        self.receiver_indices = torch.from_numpy(receiver_indices)
        self.receiver_weights = torch.from_numpy(receiver_weights)

        # Get indices of grid coords, appeared either in source_indices or receiver_indices
        used_source_mask = np.bincount(source_indices.ravel(), minlength=self.grid.n_coords) > 0
        used_receiver_mask = np.bincount(receiver_indices.ravel(), minlength=self.grid.n_coords) > 0
        used_coords_mask = used_source_mask | used_receiver_mask
        used_coords_indices = np.require(np.nonzero(used_coords_mask)[0], dtype=np.int32)
        self.used_coords_mask = torch.from_numpy(used_coords_mask)

        # Construct an interpolation grid for mean slowness estimation and get interpolation indices and weights for
        # each trace
        indices, weights = self._get_slowness_averaging_params(source_coords, receiver_coords,
                                                               grid[used_coords_indices], slowness_grid_size)
        self.mean_slowness_indices = torch.from_numpy(used_coords_indices[indices])
        self.mean_slowness_weights = torch.from_numpy(weights)

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

    @staticmethod
    def _get_sensor_interpolation_params(sensor_coords, grid):
        coords_df = pl.from_numpy(sensor_coords, schema=["X", "Y"])
        unique_coords_df = coords_df.unique()
        inverse_df = coords_df.join(unique_coords_df.with_row_count(name="i"), how="left", on=["X", "Y"])

        unique_coords = unique_coords_df.to_numpy()
        inverse = inverse_df.get_column("i").to_numpy()

        indices, weights = grid.get_interpolation_params(unique_coords)
        indices = np.require(indices, dtype=np.int32)[inverse]
        weights = np.require(weights, dtype=np.float32)[inverse]
        return indices, weights

    @staticmethod
    @njit(nogil=True, parallel=True)
    def _get_intermediate_averaging_params(source_coords, receiver_coords, grid_size):
        min_x = min(source_coords[:, 0].min(), receiver_coords[:, 0].min())
        max_x = max(source_coords[:, 0].max(), receiver_coords[:, 0].max())
        min_y = min(source_coords[:, 1].min(), receiver_coords[:, 1].min())
        max_y = max(source_coords[:, 1].max(), receiver_coords[:, 1].max())

        grid_size_x = 1 + math.ceil((max_x - min_x) / grid_size)
        grid_size_y = 1 + math.ceil((max_y - min_y) / grid_size)
        grid_coords_x = min_x + grid_size * np.arange(grid_size_x)
        grid_coords_y = min_y + grid_size * np.arange(grid_size_y)

        coords = np.empty((grid_size_x * grid_size_y, 2))
        for i in prange(grid_size_x):
            for j in range(grid_size_y):
                coords_ix = i * grid_size_y + j
                coords[coords_ix, 0] = grid_coords_x[i]
                coords[coords_ix, 1] = grid_coords_y[j]

        n_traces = len(source_coords)
        trace_n_cells = np.empty(n_traces, dtype=np.int32)
        for i in prange(n_traces):
            offset = np.sqrt(np.sum((receiver_coords[i] - source_coords[i])**2))
            trace_n_cells[i] = 1 + math.ceil(offset / grid_size)
        max_n_cells = trace_n_cells.max()

        keep_coords_mask = np.zeros(n_traces, dtype=np.bool_)
        indices = np.zeros((n_traces, max_n_cells), dtype=np.int32)
        weights = np.zeros((n_traces, max_n_cells), dtype=np.float32)

        for i in prange(n_traces):
            n_cells = trace_n_cells[i]

            coords_x = np.linspace(source_coords[i, 0], receiver_coords[i, 0], n_cells)
            indices_x = np.round((coords_x - min_x) / grid_size).astype(np.int32)
            indices_x = np.clip(indices_x, 0, grid_size_x - 1)

            coords_y = np.linspace(source_coords[i, 1], receiver_coords[i, 1], n_cells)
            indices_y = np.round((coords_y - min_y) / grid_size).astype(np.int32)
            indices_y = np.clip(indices_y, 0, grid_size_y - 1)

            intermediate_indices = indices_x * grid_size_y + indices_y
            keep_coords_mask[intermediate_indices] = True
            indices[i, :n_cells] = intermediate_indices
            weights[i, :n_cells] = 1 / n_cells

        indices_shift = np.cumsum(~keep_coords_mask)
        for i in prange(n_traces):
            n_cells = trace_n_cells[i]
            indices[i, :n_cells] -= indices_shift[indices[i, :n_cells]]

        return coords[keep_coords_mask], indices, weights

    @staticmethod
    @njit(nogil=True, parallel=True)
    def _combine_averaging_params(igrid_averaging_indices, igrid_to_grid_indices,
                                  igrid_averaging_weights, igrid_to_grid_weights):
        n_traces = len(igrid_averaging_indices)
        n_indices_per_trace = igrid_averaging_indices.shape[1] * igrid_to_grid_indices.shape[1]
        indices = np.empty((n_traces, n_indices_per_trace), dtype=np.int32)
        weights = np.empty((n_traces, n_indices_per_trace), dtype=np.float32)

        for i in prange(n_traces):
            trace_igrid_indices = igrid_averaging_indices[i]
            trace_igrid_weights = igrid_averaging_weights[i].reshape(-1, 1)

            indices[i] = igrid_to_grid_indices[trace_igrid_indices].ravel()
            weights[i] = (igrid_to_grid_weights[trace_igrid_indices] * trace_igrid_weights).ravel()

        return indices, weights

    @classmethod
    def _get_slowness_averaging_params(cls, source_coords, receiver_coords, grid, slowness_grid_size):
        # Construct an intermediate regular grid for faster neighbors search (igrid)
        igrid_params = cls._get_intermediate_averaging_params(source_coords, receiver_coords, slowness_grid_size)
        igrid_coords, igrid_averaging_indices, igrid_averaging_weights = igrid_params

        # Get interpolation params for each coordinate of the intermediate grid
        igrid_to_grid_indices, igrid_to_grid_weights = grid.get_interpolation_params(igrid_coords)

        # Get parameters of complex interpolation: coords -> intermediate grid -> final grid
        indices, weights = cls._combine_averaging_params(igrid_averaging_indices, igrid_to_grid_indices,
                                                         igrid_averaging_weights, igrid_to_grid_weights)
        return indices, weights

    # Loader creation

    def create_train_loader(self, batch_size, n_epochs, shuffle=True, drop_last=True, device=None, bar=True):
        train_tensors = [self.source_coords, self.source_elevations, self.source_indices, self.source_weights,
                         self.receiver_coords, self.receiver_elevations, self.receiver_indices, self.receiver_weights,
                         self.mean_slowness_indices, self.mean_slowness_weights, self.target_traveltimes]
        loader = TensorDataLoader(*train_tensors, batch_size=batch_size, n_epochs=n_epochs,
                                  shuffle=shuffle, drop_last=drop_last, device=device)
        return tqdm(loader, desc="Iterations of model fitting", disable=not bar)

    def create_predict_loader(self, batch_size, n_epochs=1, shuffle=False, drop_last=False, device=None, bar=True):
        pred_tensors = [self.source_coords, self.source_elevations, self.source_indices, self.source_weights,
                        self.receiver_coords, self.receiver_elevations, self.receiver_indices, self.receiver_weights,
                        self.mean_slowness_indices, self.mean_slowness_weights, self.traveltime_corrections]
        loader = TensorDataLoader(*pred_tensors, batch_size=batch_size, n_epochs=n_epochs,
                                  shuffle=shuffle, drop_last=drop_last, device=device)
        return tqdm(loader, desc="Iterations of model inference", disable=not bar)

    # Evaluation of predictions

    def evaluate(self):
        if not self.has_predictions:
            raise ValueError
        return torch.abs(self.pred_traveltimes - self.true_traveltimes).mean().item()

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
        qc_df[["SourceX", "SourceY"]] = self.source_coords.numpy()
        qc_df[["GroupX", "GroupY"]] = self.receiver_coords.numpy()
        qc_df["True"] = self.true_traveltimes.numpy()
        qc_df["Pred"] = self.pred_traveltimes.numpy()

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
