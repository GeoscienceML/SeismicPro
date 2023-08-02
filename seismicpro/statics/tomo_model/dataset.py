import numpy as np
import polars as pl

from ..dataset import TravelTimeDataset
from ...const import HDR_FIRST_BREAK


class TomoModelTravelTimeDataset(TravelTimeDataset):
    def __init__(self, survey, grid, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto"):
        super().__init__(survey, grid, first_breaks_header=first_breaks_header,
                         uphole_correction_method=uphole_correction_method)

        columns_dict = {
            "SourceSurfaceElevation": self.source_elevations,
            "SourceX": self.source_coords[:, 0],
            "SourceY": self.source_coords[:, 1],
            "ReceiverGroupElevation": self.receiver_elevations,
            "GroupX": self.receiver_coords[:, 0],
            "GroupY": self.receiver_coords[:, 1],
            "TargetTravelTime": self.target_traveltimes,
        }
        dataset_df = pl.from_dict(columns_dict).with_row_count("ix")
        gather_df_list = dataset_df.partition_by("SourceSurfaceElevation", "SourceX", "SourceY", maintain_order=False)

        gather_data_list = []
        for df in gather_df_list:
            source = df.select("SourceSurfaceElevation", "SourceX", "SourceY").row(0)
            source = np.require(source, requirements="C", dtype=np.float64)
            receivers = df.select("ReceiverGroupElevation", "GroupX", "GroupY").to_numpy()
            receivers = np.require(receivers, requirements="C", dtype=np.float64)
            target_tt = df.get_column("TargetTravelTime").to_numpy()
            dataset_pos = df.get_column("ix").to_numpy()
            gather_data_list.append((source, receivers, target_tt, dataset_pos))
        self.gather_data = gather_data_list

    @property
    def n_train_gathers(self):
        return len(self.gather_data)