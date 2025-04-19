"""Script to retrieve and plot sea surface height (SSH) data from Copernicus Marine Service."""

import datetime
import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import copernicusmarine
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mpl.use("agg")  # Use a non-interactive backend for rendering


def add_colorbar(
    fig: mpl.figure.Figure,
    ax: mpl.axes,
    var: str,
    fmt: str | None = None,
    range_limit: list | None = None,
) -> mpl.colorbar.Colorbar:
    """Add color bar to the figure and format it."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
    cb = fig.colorbar(var, cax=cax, format=fmt)
    if range_limit:
        cb.mappable.set_clim(range_limit)
    cb.ax.tick_params(which="major", labelsize=6, length=3, width=0.5, pad=0.05)
    return cb


def download_ssh_data(day: datetime, lon: list, lat: list) -> None:
    """Load SSH data from Copernicus Marine Service.

    Copernicus Marine Service (2023). Global Ocean - Sea Level - All Satellites
    https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046/description

    Args:
        day (datetime): today's date
        lon (list): longitude range
        lat (list): latitude range

    """
    next_day = day + datetime.timedelta(days=1)
    outfile = f"{day.year}-{day.month:02d}-{day.day:02d}.nc"

    copernicusmarine.subset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        variables=["adt", "ugos", "vgos"],
        minimum_longitude=lon[0],
        maximum_longitude=lon[1],
        minimum_latitude=lat[0],
        maximum_latitude=lat[1],
        start_datetime=day.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=next_day.strftime("%Y-%m-%dT%H:%M:%S"),
        minimum_depth=0,
        maximum_depth=0,
        output_filename=outfile,
        output_directory="./",
    )


def plot_gom_figure(day: datetime, lon: list, lat: list) -> None:
    """Create a figure with SSH data.

    Args:
        day (datetime): today's date
        lon (list): longitude range
        lat (list): latitude range

    """
    # load the data
    data_file = f"{day.strftime('%Y-%m-%d')}.nc"
    ds = xr.open_dataset(data_file)

    # map settings
    plot_crs = ccrs.PlateCarree()
    ticks = [4, 4]
    contour_value = 0.55

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=plot_crs)

    ax.set_title(f"{day.strftime('%Y-%m-%d')}")

    # ssh
    pcm = ax.pcolormesh(
        np.mod(ds.longitude, 180) - 180,
        ds.latitude,
        ds.isel(time=0).adt,
        transform=ccrs.PlateCarree(),
        cmap=cmocean.cm.balance,
        shading="gouraud",
    )
    cb = add_colorbar(fig, ax, pcm)
    cb.set_label("Sea Surface Height [m]", fontsize=8)

    # isoline
    ax.contour(
        np.mod(ds.longitude, 180) - 180,
        ds.latitude,
        ds.isel(time=0).adt,
        [contour_value],
        colors="black",
        linestyles="dashed",
        linewidths=1,
    )
    ax.legend(
        handles=[mlines.Line2D([], [], linestyle="dashed", color="black", label="LC isoline")],
        loc="upper left",
    )

    # map
    ax.add_feature(cfeature.LAND, facecolor="dimgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.5)

    if ax.projection == ccrs.PlateCarree():
        ax.set_xticks(np.arange(lon[0], lon[1] + 1e-6, ticks[0]), crs=plot_crs)
        ax.set_yticks(np.arange(lat[0], lat[1], ticks[1]), crs=plot_crs)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xlim(lon)
        ax.set_ylim(lat)

    fig.savefig("docs/latest.png", dpi=600)


if __name__ == "__main__":
    # must set copernicus username and password as environment variables
    copernicusmarine.login(
        username=os.getenv("COPERNICUS_USER"),
        password=os.getenv("COPERNICUS_PASS"),
        force_overwrite=True,
    )

    # GoM region
    lon = [-98, -78]
    lat = [18, 31]

    day = datetime.datetime.now(datetime.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    data_file = f"{day.strftime('%Y-%m-%d')}.nc"

    if not Path(data_file).exists():
        logger.info("Downloading data")
        args = download_ssh_data(day, lon, lat)
    if Path(data_file).exists():
        logger.info("Plotting")
        plot_gom_figure(day, lon, lat)
    else:
        logger.error("Data file does not exist.")
