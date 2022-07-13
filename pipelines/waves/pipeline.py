import xarray as xr

import matplotlib.pyplot as plt
from tsdat import IngestPipeline, get_start_date_and_time_str, get_filename

from utils import format_time_xticks


class Waves(IngestPipeline):
    """--------------------------------------------------------------------------------
    SPOTTER_BUOY INGESTION PIPELINE

    Wave data taken in Clallam Bay over a month-long deployment in Aug-Sep 2021

    --------------------------------------------------------------------------------"""

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        ds = dataset
        location = self.dataset_config.attrs.location_id
        datastream: str = self.dataset_config.attrs.datastream

        date, time = get_start_date_and_time_str(dataset)

        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        with self.storage.uploadable_dir(datastream) as tmp_dir:
            fig, ax = plt.subplots()

            ax.plot(ds.time, ds.displacement.sel(dir="x"), label="x-direction")
            ax.plot(ds.time, ds.displacement.sel(dir="y"), label="y-direction")
            ax.plot(ds.time, ds.displacement.sel(dir="z"), label="z-direction")

            ax.set_title("")  # Remove bogus title created by xarray
            ax.legend(ncol=2, bbox_to_anchor=(1, -0.05))
            ax.set_ylabel("Buoy Displacement [m]")
            ax.set_xlabel("Time [UTC]")
            # format_time_xticks(ax, date_format="%Y-%m-%d %H:%M")
            plt.legend()

            plot_file = get_filename(
                dataset, title="buoy_displacement", extension="png"
            )
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)
        pass
