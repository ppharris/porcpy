import iris
import iris.analysis.stats
import iris.plot as iplt
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings

from esmvaltool.diag_scripts.shared import (
    run_diagnostic, group_metadata, select_metadata, get_plot_filename)

np.seterr(divide='ignore', invalid='ignore')


def _get_plot_filename(var_meta, cfg):
    """Return an output filename for TCI map plots."""
    basename = "_".join([var_meta["project"],
                         var_meta["dataset"],
                         var_meta["exp"],
                         var_meta["ensemble"],
                         "tci"])

    filename = get_plot_filename(basename, cfg)
    return filename


def make_tci(sm, flux, conf=99.):
    """Calculate terrestrial coupling index for each gridcell.

    The coupling index is defined in Dirmeyer (2011) as
    tci=covar(SM,FLUX)/std(SM)

    flux: flux time series (either latent or sensible heat)
    sm: soil moisture time series
    conf: threshold of statistical confidence for the correlation
          between sm and flux. It should be in the range [0,100].
          Grid-cells with not reach the confidence threshold will be
          masked.  If conf=0 the statistical significance is not
          computed.
    """

    # Calculation of coupling index. The covar is computed from correlation as
    # there isn't covar function on Iris:
    #
    # TCI = slope(SM, HF) * std(SM)
    #     = cov(SM, HF) / std(SM)
    #     = pearsonr(SM, HF) * std(HF)
    pearson = iris.analysis.stats.pearsonr(sm, flux, corr_coords='time')
    s_hf = flux.collapsed('time', iris.analysis.STD_DEV)

    tci = pearson * s_hf

    # Masking out gridcells with correlation statistically not significant
    if (conf > 0):
        p_thre = 1. - conf/100.
        for la in range(tci.shape[0]):
            for lo in range(tci.shape[1]):
                if (~tci.data.mask[la, lo]):
                    rho = stats.pearsonr(sm.data[:, la, lo],
                                         flux.data[:, la, lo])
                    if rho[1] > p_thre:
                        tci.data.mask[la, lo] = True
    return tci


def make_diag_tci(cfg, dataset, input_data,
                  sm_name="mrlsl", hf_name="hfls"):

    sm_meta = select_metadata(input_data, short_name=sm_name)[0]
    hf_meta = select_metadata(input_data, short_name=hf_name)[0]

    # Load variables
    sm = iris.load_cube(sm_meta["filename"])
    hf = iris.load_cube(hf_meta["filename"])

    # Split by seasons
    iris.coord_categorisation.add_season(sm, 'time')
    smmam = sm.extract(iris.Constraint(season='mam'))
    smjja = sm.extract(iris.Constraint(season='jja'))
    smson = sm.extract(iris.Constraint(season='son'))
    smdjf = sm.extract(iris.Constraint(season='djf'))

    iris.coord_categorisation.add_season(hf, 'time')
    hfmam = hf.extract(iris.Constraint(season='mam'))
    hfjja = hf.extract(iris.Constraint(season='jja'))
    hfson = hf.extract(iris.Constraint(season='son'))
    hfdjf = hf.extract(iris.Constraint(season='djf'))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message=".*Collapsing.*",
            category=UserWarning,
        )
        # Compute tci with statistical significance
        tcmam = make_tci(smmam, hfmam)
        tcjja = make_tci(smjja, hfjja)
        tcson = make_tci(smson, hfson)
        tcdjf = make_tci(smdjf, hfdjf)

    # Plot results
    fig = plt.figure(figsize=(9, 5), dpi=100)
    levels = np.linspace(-40, 40, 17)
    cmap = 'jet'

    ax = plt.subplot(2, 2, 1)
    cmam = iplt.contourf(tcmam, cmap=cmap, levels=levels)
    plt.gca().coastlines()
    plt.title('MAM')

    plt.subplot(2, 2, 2)
    cjja = iplt.contourf(tcjja, cmap=cmap, levels=levels)
    plt.gca().coastlines()
    plt.title('JJA')

    plt.subplot(2, 2, 3)
    cson = iplt.contourf(tcson, cmap=cmap, levels=levels)
    plt.gca().coastlines()
    plt.title('SON')

    plt.subplot(2, 2, 4)
    cdjf = iplt.contourf(tcdjf, cmap=cmap, levels=levels)
    plt.gca().coastlines()
    plt.title('DJF')

    title = ("Terrestrial Coupling Index "
             "({start_year}-{end_year}) {sm_name} - {hf_name}".format(
                 start_year=sm_meta["start_year"],
                 end_year=sm_meta["end_year"],
                 sm_name=sm_name,
                 hf_name=hf_name,
             ))

    fig.suptitle(title)
    cax = plt.axes([0.15, 0.05, 0.72, 0.03])
    fig.colorbar(cmam, orientation='horizontal', cax=cax)

    if cfg["write_plots"]:
        plt.savefig(_get_plot_filename(sm_meta, cfg))

    return None


def main(cfg):
    grouped_metadata = group_metadata(cfg["input_data"].values(), "dataset")

    for dataset, data in grouped_metadata.items():
        file_tci = make_diag_tci(cfg, dataset, data)


if __name__ == '__main__':
    with run_diagnostic() as cfg:
        main(cfg)
