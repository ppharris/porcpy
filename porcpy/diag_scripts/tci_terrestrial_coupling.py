import cartopy.crs as ccrs
import iris
import iris.analysis.stats as istats
import iris.plot as iplt
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sstats
import warnings

from esmvaltool.diag_scripts.shared import (run_diagnostic,
                                            group_metadata,
                                            select_metadata,
                                            get_diagnostic_filename,
                                            get_plot_filename)


np.seterr(divide='ignore', invalid='ignore')


def calc_pearsonr(cubex, cubey, corr_coords, alpha=None):
    """Calculate the Pearson's r correlation coefficient with significance.

    Calculates Pearson's r over the specified coordinates of two Iris cubes.
    These cubes must have the same coordinates, but this is not checked.  A
    common data mask on cubex and cubey is enforced when calculating r.
    If alpha is set, p-values are calculated following the notes in,

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    which are used to mask boxes with pval > alpha.

    Parameters
    ----------
    cubex, cubey: iris.cube.Cube
        Input data cubes to be correlated.
    corr_coords: str or list of str
        The cube coordinate name(s) over which to calculate correlations.
    alpha : float, optional
        Type I error rate above which r values are rejected, range (0, 1).

    Returns
    -------
    corr: iris.cube.Cube
        Pearson's r correlation coefficient.

    """

    corr = istats.pearsonr(cubex, cubey, corr_coords, common_mask=True)

    if alpha is not None:
        # Calculate the N in the correlations for each gridbox.  This seems
        # long-winded for such a common calculation, is there a better way of
        # doing it?
        common_mask = (cubex.data.mask | cubey.data.mask)

        counter = cubex.copy()
        counter.data = np.ma.where(common_mask, 0, 1)
        counter.data.mask = common_mask
        count = counter.collapsed(corr_coords, iris.analysis.COUNT,
                                  function=lambda x: x)

        # Calculate the p-value of the correlation from r and N values.
        rabs = -np.ma.abs(corr.data)
        shps = count.data/2. - 1
        pval = 2 * sstats.beta.cdf(rabs, shps, shps, loc=-1, scale=2)

        # Mask out correlations (1) made from two value or fewer or (2) with
        # p-values greater than the requested alpha level.
        extra_mask = (count.data <= 2) | (pval > alpha)
        corr.data.mask[extra_mask] = True

    return corr


def calc_tci(sm, flux, tci_attrs=None, alpha=None):
    """Calculate terrestrial coupling index for each grid box.

    The coupling index is defined in Dirmeyer (2011) GRL,
    https://doi.org/10.1029/2011GL048268:

    TCI = slope(SM, HF) * std(SM)
        = cov(SM, HF) / std(SM)
        = pearsonr(SM, HF) * std(HF)

    This code uses the third form of TCI above because Iris provides pearsonr
    but not a covariance function.  This is a simple version that pools all
    available time steps within a season, across all years, so there is no
    separation by intra-seasonal and inter-annual variability.

    Parameters
    ----------
    sm : iris.cube.Cube
        Cube of soil moisture time series.  This must have auxiliary coord
        "season", e.g., as added by iris.coord_categorisation.add_season().
    flux : iris.cube.Cube
        Cube of surface heat flux time series (e.g., latent or sensible heat
        flux).  Must also have "season" aux_coord.
    tci_attrs : dict, optional
        A dictionary of attributes to be added to the output Cube.
    alpha : float, optional
        Type I error rate above which r values are rejected, range (0, 1).

    Returns
    -------
    tci : iris.cube.Cube
        Cube of terrestrial coupling index by season.

    """

    # Calculation of coupling index.
    agg_coord = "season"

    s_hf = flux.aggregated_by(agg_coord, iris.analysis.STD_DEV)

    pearson = iris.cube.CubeList()
    for season in s_hf.coord(agg_coord).points:
        sm_seas = sm.extract(iris.Constraint(season=season))
        flux_seas = flux.extract(iris.Constraint(season=season))
        pearson.append(
            calc_pearsonr(sm_seas, flux_seas, corr_coords=agg_coord,
                          alpha=alpha)
        )

    pearson = pearson.merge_cube()
    pearson.replace_coord(s_hf.coord(agg_coord))

    tci = pearson * s_hf

    for kv in tci_attrs.items():
        tci.__setattr__(*kv)

    return tci


def make_tci(filename_sm, filename_hf, filename_tci,
             standard_name_sm="moisture_content_of_soil_layer",
             standard_name_hf="surface_upward_latent_heat_flux",
             standard_name_tci="terrestrial_coupling_index",
             varname_tci="tci",
             alpha=None):
    """Driver routine for calculating a terrestrial coupling index (TCI).

    Parameters
    ----------
    filename_sm : str
        Name of input netCDF file containing soil moisture time series.
    filename_hf : str
        Name of input netCDF file containing surface heat flux time series,
        e.g., sensible of latent heat.
    filename_tci : str
        Name of output netCDF file containing terrestrial coupling index for a
        selection of calendar seasons.  Any existing file with the same name
        will be clobbered.
    standard_name_sm : str, optional
        NetCDF standard_name for input soil moisture data.
    standard_name_hf : str, optional
        NetCDF standard_name name for input surface heat flux data.
    standard_name_tci : str, optional
       NetCDF standard_name for output TCI data.
    varname_tci : str, optional
        NetCDF variable name for output TCI data.
    alpha : float, optional
        Type I error rate (p-value) above which correlations are rejected in
        the calculation of TCI.  Values in the range (0, 1).

    Returns
    -------
    tci : iris.cube.Cube
        Terrestrial coupling index for a selection of seasons.

    """

    # Load variables
    sm = iris.load_cube(filename_sm, standard_name_sm)
    hf = iris.load_cube(filename_hf, standard_name_hf)
    iris.coord_categorisation.add_season(sm, 'time')
    iris.coord_categorisation.add_season(hf, 'time')

    # NB There is no official standard_name for TCI, so use the long_name.
    tci_attrs = {
        "var_name": varname_tci,
        "long_name": standard_name_tci,
        "units": "W m-2",
    }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message=".*Collapsing.*",
            category=UserWarning,
        )
        tci = calc_tci(sm, hf, tci_attrs=tci_attrs, alpha=alpha)

    # Write out TCI to netCDF.
    iris.save(tci, filename_tci)

    return tci


def plot_tci(data_tci, file_out_plot, title=None):
    """Output a plot of seasonal maps of terrestrial coupling index.

    Parameters
    ----------
    data_tci : iris.cube.Cube
        Cube of terrestrial coupling index to be plotted.  This must have the
        auxiliary coord "season", e.g., as added by
        iris.coord_categorisation.add_season().
    file_out_plot : str
        Output image file name.  Any existing file with this name will be
        clobbered.
    title : str, optional
        Title to be added to the overall plot as a fig.suptitle().  If absent,
        a basic title is added instead.

    """

    if title is None:
        title = "Terrestrial coupling index"

    levels = np.linspace(-40, 40, 17)
    cmap = 'jet'

    # Plot results
    F, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 5), dpi=100,
                          subplot_kw=dict(projection=ccrs.PlateCarree()))

    for ax, cube in zip(axs.flat, data_tci.slices_over("season")):
        CF = iplt.contourf(cube, axes=ax, cmap=cmap, levels=levels)
        ax.coastlines()
        ax.set_title(cube.coord("season").points[0].upper())

    F.suptitle(title)
    cax = plt.axes([0.15, 0.05, 0.72, 0.03])
    F.colorbar(CF, orientation='horizontal', cax=cax)

    plt.savefig(file_out_plot)

    return


###############################################################################
# Code below this point is ESMValTool-aware, i.e., it may reference the
# ESMValTool configuration dictionary.
###############################################################################


def _get_filename(var_meta, cfg, extension="nc"):
    """Return a filename for output data, e.g., coupling index."""
    basename = "_".join([
        var_meta["project"],
        var_meta["dataset"],
        var_meta["exp"],
        var_meta["ensemble"],
        var_meta["short_name"],
        str(var_meta["start_year"]),
        str(var_meta["end_year"]),
    ])

    filename = get_diagnostic_filename(basename, cfg, extension=extension)
    return filename


def _get_plot_filename(var_meta, cfg, label):
    """Return an output filename for plots."""
    basename = "_".join([
        var_meta["project"],
        var_meta["dataset"],
        var_meta["exp"],
        var_meta["ensemble"],
        label,
        str(var_meta["start_year"]),
        str(var_meta["end_year"]),
    ])

    filename = get_plot_filename(basename, cfg)
    return filename


def make_diag_tci(cfg, dataset, input_data,
                  sm_name="mrlsl", hf_name="hfls", tci_name="tci"):
    """Shim routine between ESMValTool and the generic make_tci()."""

    sm_meta = select_metadata(input_data, short_name=sm_name)[0]
    hf_meta = select_metadata(input_data, short_name=hf_name)[0]

    sm_meta["standard_name"] = "depth_integrated_moisture_content_of_soil_layer"  # noqa

    tci_meta = sm_meta.copy()
    tci_meta["short_name"] = tci_name
    tci_meta["standard_name"] = "terrestrial_coupling_index"

    filename_tci = _get_filename(tci_meta, cfg, extension="nc")

    alpha = 0.05  # pvalue rejection threshold.

    tci = make_tci(sm_meta["filename"], hf_meta["filename"], filename_tci,
                   standard_name_sm=sm_meta["standard_name"],
                   standard_name_hf=hf_meta["standard_name"],
                   standard_name_tci=tci_meta["standard_name"],
                   alpha=alpha)

    return tci


def make_plots(cfg, dataset, data, data_tci,
               varname_sm="mrlsl",
               varname_hf="hfls",
               varname_tci="tci"):
    """Shim routine between ESMValTool and the generic plot_tci()."""

    meta = select_metadata(data, short_name=varname_sm)[0]

    filename_maps = _get_plot_filename(meta, cfg, varname_tci)

    model_desc = "{:s}, {:s}, {:s}, {:s}, {:d}-{:d}".format(
        meta["project"],
        meta["exp"],
        meta["dataset"],
        meta["ensemble"],
        meta["start_year"],
        meta["end_year"],
    )

    title = ("Terrestrial Coupling Index "
             "({units}) {varname_sm} - {varname_hf}\n{model_desc}".format(
                 units=str(data_tci.units),
                 varname_sm=varname_sm,
                 varname_hf=varname_hf,
                 model_desc=model_desc,
             ))

    plot_tci(data_tci, filename_maps, title=title)
    return


def main(cfg):
    grouped_metadata = group_metadata(cfg["input_data"].values(), "dataset")

    for dataset, data in grouped_metadata.items():
        data_tci = make_diag_tci(cfg, dataset, data)
        if cfg["write_plots"]:
            make_plots(cfg, dataset, data, data_tci)
    return


if __name__ == '__main__':
    with run_diagnostic() as cfg:
        main(cfg)
