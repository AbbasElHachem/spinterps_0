# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

test simple kriging cython class
"""
import os
os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import pyximport
import numpy as np
# pyximport.install()
pyximport.install()

from spinterps import (OrdinaryKriging)

import timeit
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr as spearmanr
from scipy.stats import pearsonr as pearsonr
from pathlib import Path
import shapefile

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})

# =============================================================================

main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
os.chdir(main_dir)

out_plots_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

path_to_data = main_dir / r'oridinary_kriging_compare_DWD_Netatmo/needed_dfs'

path_netatmo_extremes_df = path_to_data / r'netatmo_daily_maximum_100_days.csv'
path_dwd_extremes_df = path_to_data / r'dwd_daily_maximum_100_days.csv'

path_to_dwd_daily_vgs = path_to_data / r'vgs_strs_dwd_daily_ppt_.csv'

path_to_dwd_daily_data = (path_to_data /
                          r'all_dwd_daily_ppt_data_combined_2014_2019_.csv')

path_to_netatmo_daily_data = path_to_data / r'all_netatmo_ppt_data_daily_.csv'

path_to_dwd_daily_edf = (path_to_data /
                         r'edf_ppt_all_dwd_daily_all_stns_combined_.csv')

path_to_netatmo_daily_edf = (path_to_data /
                             r'edf_ppt_all_netamo_daily_all_stns_combined_.csv')


path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

path_to_netatmo_coords = path_to_data / r'netatmo_bw_1hour_coords_utm32.csv'

path_tp_netatmo_gd_stns = (path_to_data /
                           r'keep_stns_all_neighbor_92_per_60min_.csv')

path_to_shpfile = (
    r"X:\exchange\ElHachem\Netatmo\Landesgrenze_ETRS89\Landesgrenze_10000_ETRS89_lon_lat.shp")

# =============================================================================
# random points and values
n_pts = 160
n_pts_nebs = 63
xi = -5 + 10 * np.random.random(n_pts_nebs)
yi = -5 + 10 * np.random.random(n_pts_nebs)
zi = -50 + 100 * np.random.random(n_pts_nebs)
# zi = np.array([1., 1., 1., 1., 1.])
# si = a + b * zi
si = -50 + 100 * np.random.random(n_pts_nebs)

xk = -5 + 10 * np.random.random(int(n_pts * 0.1))
yk = -5 + 10 * np.random.random(int(n_pts * 0.1))
# sk = -5 + 10 * np.random.random(int(n_pts * 0.1))
#
# n_drifts = 1
# si_md = -50 + 100 * np.random.random((n_drifts, n_pts_nebs))
# sk_md = -5 + 10 * np.random.random((n_drifts, int(n_pts * 0.1)))
# =============================================================================

strt_date = '2014-01-01'
end_date = '2019-08-01'

use_netatmo_stns_for_kriging = True
use_dwd_stns_for_kriging = False

normal_kriging = False
qunatile_kriging = True

cross_validate = False

use_netatmo_gd_stns = True
# =============================================================================
if normal_kriging:
    netatmo_data_to_use = path_to_netatmo_daily_data
    dwd_data_to_use = path_to_dwd_daily_data
    plot_unit = r'(mm_per_day)'
    title_ = r'Amounts'

if qunatile_kriging:
    netatmo_data_to_use = path_to_netatmo_daily_edf
    dwd_data_to_use = path_to_dwd_daily_edf
    plot_unit = 'CDF'
    title_ = r'Quantiles'

#==============================================================================
# # Netatmo DATA and COORDS
#==============================================================================
netatmo_in_coords_df = pd.read_csv(path_to_netatmo_coords,
                                   index_col=0,
                                   sep=';',
                                   encoding='utf-8')

netatmo_in_vals_df = pd.read_csv(
    netatmo_data_to_use, sep=';',
    index_col=0,
    encoding='utf-8', engine='c')

netatmo_in_vals_df.index = pd.to_datetime(
    netatmo_in_vals_df.index, format='%Y-%m-%d')

netatmo_in_vals_df = netatmo_in_vals_df.loc[strt_date:end_date, :]
netatmo_in_vals_df.dropna(how='all', axis=0, inplace=True)
# daily sums
netatmo_in_vals_df = netatmo_in_vals_df[(0 <= netatmo_in_vals_df) &
                                        (netatmo_in_vals_df <= 300)]

cmn_stns = netatmo_in_coords_df.index.intersection(netatmo_in_vals_df.columns)
netatmo_in_vals_df = netatmo_in_vals_df.loc[:, cmn_stns]
#==============================================================================

if use_netatmo_gd_stns:
    path_to_netatmo_gd_stns = path_to_data / \
        r'keep_stns_all_neighbor_92_per_60min_.csv'
    df_gd_stns = pd.read_csv(path_to_netatmo_gd_stns,
                             index_col=0,
                             sep=';',
                             encoding='utf-8')
    good_netatmo_stns = df_gd_stns.loc[:, 'Stations'].values.ravel()
    in_vals_df = netatmo_in_vals_df.loc[:, good_netatmo_stns]
    netatmo_in_coords_df = netatmo_in_coords_df.loc[good_netatmo_stns, :]
    cmn_stns = netatmo_in_coords_df.index.intersection(
        netatmo_in_vals_df.columns)
    netatmo_in_vals_df = netatmo_in_vals_df.loc[:, cmn_stns]
    title_ = title_ + r'_using_Netatmo_good_stations_'
#

#==============================================================================
# # DWD DATA AND COORDS
#==============================================================================
dwd_in_coords_df = pd.read_csv(path_to_dwd_coords,
                               index_col=0,
                               sep=';',
                               encoding='utf-8')

dwd_in_vals_df = pd.read_csv(
    dwd_data_to_use, sep=';', index_col=0, encoding='utf-8')
dwd_in_vals_df.index = pd.to_datetime(
    dwd_in_vals_df.index, format='%Y-%m-%d')

dwd_in_vals_df = dwd_in_vals_df.loc[strt_date:end_date, :]
dwd_in_vals_df.dropna(how='all', axis=0, inplace=True)
# daily sums
dwd_in_vals_df = dwd_in_vals_df[(0 <= dwd_in_vals_df) &
                                (dwd_in_vals_df <= 300)]

# added by Abbas, for DWD stations
stndwd_ix = ['0' * (5 - len(str(stn_id))) + str(stn_id)
             if len(str(stn_id)) < 5 else str(stn_id)
             for stn_id in dwd_in_coords_df.index]

dwd_in_coords_df.index = stndwd_ix
dwd_in_coords_df.index = list(map(str, dwd_in_coords_df.index))


#==============================================================================
# # NETATMO AND DWD EXTREME EVENTS
#==============================================================================
netatmo_in_extremes_df = pd.read_csv(path_netatmo_extremes_df,
                                     index_col=0,
                                     sep=';',
                                     encoding='utf-8')

dwd_in_extremes_df = pd.read_csv(path_dwd_extremes_df,
                                 index_col=0,
                                 sep=';',
                                 encoding='utf-8')

#==============================================================================
# # VG MODELS
#==============================================================================
df_vgs = pd.read_csv(path_to_dwd_daily_vgs,
                     index_col=0,
                     sep=';',
                     encoding='utf-8')
df_vgs.index = pd.to_datetime(df_vgs.index, format='%Y-%m-%d')
df_vgs_models = df_vgs.iloc[:, 0]
df_vgs_models.dropna(how='all', inplace=True)

#==============================================================================
#
#==============================================================================


def _plot_interp(
        _interp_x_crds_plt_msh,
        _interp_y_crds_plt_msh,
        interp_fld,
        _index_type,
        curr_x_coords,
        curr_y_coords,
        interp_time,
        model,
        interp_type,
        out_figs_dir,
        data_vals,
        _nc_vlab,
        _nc_vunits):

    if _index_type == 'date':
        time_str = interp_time.strftime('%Y_%m_%d_T_%H_%M')

    elif _index_type == 'obj':
        time_str = interp_time

    out_fig_name = f'{interp_type.lower()}_{time_str}.png'

    fig, ax = plt.subplots()

    if not np.all(np.isfinite(interp_fld)):
        grd_min = np.nanmin(data_vals)
        grd_max = np.nanmax(data_vals)

    else:
        grd_min = interp_fld.min()
        grd_max = interp_fld.max()

    # TODO: added by Abbas for EDF plotting
    pclr = ax.pcolormesh(
        _interp_x_crds_plt_msh,
        _interp_y_crds_plt_msh,
        interp_fld,
        cmap=plt.get_cmap('viridis'),
        vmin=0,  # grd_min,
        vmax=60)  # grd_max)

    cb = fig.colorbar(pclr)

    cb.set_label(_nc_vlab + ' (' + _nc_vunits + ')')

    ax.scatter(
        curr_x_coords,
        curr_y_coords,
        label='obs. pts.',
        marker='+',
        c='r',
        alpha=0.7)

    ax.legend(framealpha=0.5)

    # read and plot shapefile (BW or Germany) should be lon lat

    shp_de = shapefile.Reader(path_to_shpfile)
    for shape_ in shp_de.shapeRecords():
        lon = [i[0] for i in shape_.shape.points[:][::-1]]
        lat = [i[1] for i in shape_.shape.points[:][::-1]]

        ax.scatter(lon, lat, marker='.', c='lightgrey',
                   alpha=0.25, s=1)

    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')

    title = (
        f'Time: {time_str}\n(VG: {model})\n'
        f'Min.: {grd_min:0.4f}, Max.: {grd_max:0.4f}')

    ax.set_title(title)

    plt.setp(ax.get_xmajorticklabels(), rotation=70)
    ax.set_aspect('equal', 'datalim')

    plt.savefig(str(out_figs_dir / out_fig_name), bbox_inches='tight')
    plt.close()
    return


print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
start = timeit.default_timer()  # to get the runtime of the program

#==============================================================================
# # ordinary kriging
#==============================================================================

for event_date, event_value in dwd_in_extremes_df.iterrows():

    if ((event_date in df_vgs_models.index) and
        (event_date in dwd_in_vals_df.index) and
            (event_date in netatmo_in_vals_df.index)):
        print('Event date is', event_date)

        vgs_model = df_vgs_models.loc[event_date]
        # DWD data and coords
        dwd_df = dwd_in_vals_df.loc[event_date, :].dropna(how='all')
        dwd_vals = dwd_df.values
        dwd_coords = dwd_in_coords_df.loc[dwd_df.index]
        x_dwd, y_dwd = dwd_coords.X.values, dwd_coords.Y.values

        # Netatmo data and coords
        netatmo_df = netatmo_in_vals_df.loc[event_date, :].dropna(how='all')
        netatmo_vals = netatmo_df.values
        netatmo_coords = netatmo_in_coords_df.loc[netatmo_df.index]
        x_netatmo, y_netatmo = netatmo_coords.X.values, netatmo_coords.Y.values

        print('\a Ordinary Kriging...')

        if use_netatmo_stns_for_kriging:
            print('using Netatmo stations to find DWD values')
            measured_vals = dwd_vals
            used_vals = netatmo_vals

            xlabel = 'DWD observed values'
            ylabel = 'DWD interpolated values using Netatmo data'
            measured_stns = 'DWD'
            used_stns = 'Netatmo'
            plot_title_acc = '_using_Netatmo_stations_to_find_DWD_values_'
            ordinary_kriging = OrdinaryKriging(
                xi=x_netatmo,
                yi=y_netatmo,
                zi=netatmo_vals,
                xk=x_dwd,
                yk=y_dwd,
                model=vgs_model)

            ordinary_kriging.krige()

        if use_dwd_stns_for_kriging:
            print('using DWD stations to find Netatmo values')
            measured_vals = netatmo_vals
            used_vals = dwd_vals

            xlabel = 'Netatmo observed values'
            ylabel = 'Netatmo interpolated values using DWD data'
            measured_stns = 'Netatmo'
            used_stns = 'DWD'
            plot_title_acc = '_using_DWD_stations_to_find_Netatmo_values_'

            ordinary_kriging = OrdinaryKriging(
                xi=x_dwd,
                yi=y_dwd,
                zi=dwd_vals,
                xk=x_netatmo,
                yk=y_netatmo,
                model=vgs_model)
            ordinary_kriging.krige()

        print('\nDistances are:\n', ordinary_kriging.in_dists)
        print('\nVariances are:\n', ordinary_kriging.in_vars)
        print('\nRight hand sides are:\n', ordinary_kriging.rhss)
        print('\nzks are:', ordinary_kriging.zk)
        print('\nest_vars are:\n', ordinary_kriging.est_vars)
        print('\nlambdas are:\n', ordinary_kriging.lambdas)
        print('\nmus are:\n', ordinary_kriging.mus)
        print('\n\n')

        spr_corr = spearmanr(measured_vals, ordinary_kriging.zk)[0]
        pear_corr = pearsonr(measured_vals, ordinary_kriging.zk)[0]

        plt.ioff()
        fig, ax = plt.subplots(figsize=(16, 12), dpi=100)

        ax.scatter(measured_vals, ordinary_kriging.zk,
                   color='r', marker='*', alpha=0.75, s=25,
                   label=(('Nbr used %s stns %d \n'
                           'Nbr used %s stns %d')
                          % (measured_stns, measured_vals.shape[0],
                             used_stns, used_vals.shape[0])))

        ax.set_xlim(
            [-0.1, max(measured_vals.max(), ordinary_kriging.zk.max()) + .2])
        ax.set_ylim(
            [-0.1, max(measured_vals.max(), ordinary_kriging.zk.max()) + .2])

        ax.plot([0, max(measured_vals.max(), ordinary_kriging.zk.max()) + .2],
                [0, max(measured_vals.max(), ordinary_kriging.zk.max()) + .2],
                color='k', alpha=0.25, linestyle='--')

        ax.set_title(('Observed vs Interpolated Rainfall %s %s '
                      '%s \n'
                      'Event Date:%s and Variogram model: %s\n'
                      'Pearson Corr: %0.2f and Spearman Corr: %0.2f'
                      % (title_, plot_unit, plot_title_acc,
                         event_date, vgs_model, pear_corr, spr_corr)))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend(loc=0)
        ax.grid(alpha=0.5)

        plt.savefig(out_plots_path /
                    (r"observed_vs_interpolated_%s_%s_%s_%s.png"
                     % (title_, plot_unit, event_date, plot_title_acc)),
                    frameon=True, papertype='a4',
                    bbox_inches='tight', pad_inches=.2)
        plt.clf()
        plt.close('all')

    else:
        print('no Variogram for this event')
        continue

# if ordinary:
#     print('\a Ordinary Kriging...')
#     for i in range(n):
#         ordinary_kriging = OrdinaryKriging(
#             xi=xi, yi=yi, zi=zi, xk=xk, yk=yk, model=model)
#         ordinary_kriging.krige()
#
#     print('\nDistances are:\n', ordinary_kriging.in_dists)
#     print('\nVariances are:\n', ordinary_kriging.in_vars)
#     print('\nRight hand sides are:\n', ordinary_kriging.rhss)
#     print('\nzks are:', ordinary_kriging.zk)
#     print('\nest_vars are:\n', ordinary_kriging.est_vars)
#     print('\nlambdas are:\n', ordinary_kriging.lambdas)
#     print('\nmus are:\n', ordinary_kriging.mus)
#     print('\n\n')

#     print('\a Ordinary Kriging with matmul...')
#     for i in range(n):
#         ordinary_kriging = kriging_02.OrdinaryKriging(xi=xi, yi=yi, zi=zi, xk=xk, yk=yk, model=model)
#         ordinary_kriging.krige()
#
#     print('\nDistances are:\n', ordinary_kriging.in_dists)
#     print('\nVariances are:\n', ordinary_kriging.in_vars)
#     print('\nRight hand sides are:\n', ordinary_kriging.rhss)
#     print('\nzks are:', ordinary_kriging.zk)
#     print('\nest_vars are:\n', ordinary_kriging.est_vars)
#     print('\nlambdas are:\n', ordinary_kriging.lambdas)
#     print('\nmus are:\n', ordinary_kriging.mus)
#     print('\n\n')

#==============================================================================
# # simple kriging
#==============================================================================
# if simple:
#     print('\a Simple Kriging...')
#     for i in range(n):
#         simple_kriging = kriging.SimpleKriging(
#             xi=xi, yi=yi, zi=zi, xk=xk, yk=yk, model=model)
#         simple_kriging.krige()
#
#     print('\nDistances are:\n', simple_kriging.in_dists)
#     print('\nVariances are:\n', simple_kriging.in_covars)
#     print('\nRight hand sides are:\n', simple_kriging.rhss)
#     print('\nzks are:', simple_kriging.zk)
#     print('\nest_vars are:\n', simple_kriging.est_covars)
#     print('\nlambdas are:\n', simple_kriging.lambdas)
#     print('\n\n')
#
# #     print('\a Simple Kriging with matmul...')
# #     for i in range(n):
# #         simple_kriging = kriging_02.SimpleKriging(xi=xi, yi=yi, zi=zi, xk=xk, yk=yk, model=model)
# #         simple_kriging.krige()
# #
# #     print('\nDistances are:\n', simple_kriging.in_dists)
# #     print('\nVariances are:\n', simple_kriging.in_covars)
# #     print('\nRight hand sides are:\n', simple_kriging.rhss)
# #     print('\nzks are:', simple_kriging.zk)
# #     print('\nest_vars are:\n', simple_kriging.est_covars)
# #     print('\nlambdas are:\n', simple_kriging.lambdas)
# #     print('\n\n')
#
# #==============================================================================
# # # external drift kriging
# #==============================================================================
# if external:
#     print('\a External Drift Kriging...')
#     for i in range(n):
#         external_drift_kriging = kriging.ExternalDriftKriging(xi=xi,
#                                                               yi=yi,
#                                                               zi=zi,
#                                                               si=si,
#                                                               xk=xk,
#                                                               yk=yk,
#                                                               sk=sk,
#                                                               model=model)
#         external_drift_kriging.krige()
#
#     print('\nDistances are:\n', external_drift_kriging.in_dists)
#     print('\nVariances are:\n', external_drift_kriging.in_vars)
#     print('\nRight hand sides are:\n', external_drift_kriging.rhss)
#     print('\nzks are:\n', external_drift_kriging.zk)
#     print('\nlambdas are:\n', external_drift_kriging.lambdas)
#     print('\nmus_1 are:\n', external_drift_kriging.mus_1)
#     print('\nmus_2 are:\n', external_drift_kriging.mus_2)
#     print('\n\n')
#
# #     print('\a External Drift Kriging with matmul...')
# #     for i in range(n):
# #         external_drift_kriging = kriging_02.ExternalDriftKriging(xi=xi,
# #                                                                 yi=yi,
# #                                                                 zi=zi,
# #                                                                 si=si,
# #                                                                 xk=xk,
# #                                                                 yk=yk,
# #                                                                 sk=sk,
# #                                                                 model=model)
# #         external_drift_kriging.krige()
# #
# #     print('\nDistances are:\n', external_drift_kriging.in_dists)
# #     print('\nVariances are:\n', external_drift_kriging.in_vars)
# #     print('\nRight hand sides are:\n', external_drift_kriging.rhss)
# #     print('\nzks are:\n', external_drift_kriging.zk)
# #     print('\nlambdas are:\n', external_drift_kriging.lambdas)
# #     print('\nmus_1 are:\n', external_drift_kriging.mus_1)
# #     print('\nmus_2 are:\n', external_drift_kriging.mus_2)
# #     print('\n\n')
#
# #==============================================================================
# # # external drift kriging with multiple drifts
# #==============================================================================
# if external_md:
#     print('\a External Drift Kriging with multiple drifts...')
#     for i in range(n):
#         external_drift_kriging_md = kriging.ExternalDriftKriging_MD(xi=xi,
#                                                                     yi=yi,
#                                                                     zi=zi,
#                                                                     si=si_md,
#                                                                     xk=xk,
#                                                                     yk=yk,
#                                                                     sk=sk_md,
#                                                                     model=model)
#         external_drift_kriging_md.krige()
#
#     print('\nDistances are:\n', external_drift_kriging_md.in_dists)
#     print('\nVariances are:\n', external_drift_kriging_md.in_vars)
#     print('\nRight hand sides are:\n', external_drift_kriging_md.rhss)
#     print('\nzks are:\n', external_drift_kriging_md.zk)
#     print('\nlambdas are:\n', external_drift_kriging_md.lambdas)
#     print('\nmus_arr is:\n', external_drift_kriging_md.mus_arr)
#     print('\n\n')
#
# #     print('\a External Drift Kriging with multiple drifts with matmul...')
# #     for i in range(n):
# #         external_drift_kriging_md = kriging_02.ExternalDriftKriging_MD(xi=xi,
# #                                                                     yi=yi,
# #                                                                     zi=zi,
# #                                                                     si=si_md,
# #                                                                     xk=xk,
# #                                                                     yk=yk,
# #                                                                     sk=sk_md,
# #                                                                     model=model)
# #         external_drift_kriging_md.krige()
# #
# #     print('\nDistances are:\n', external_drift_kriging_md.in_dists)
# #     print('\nVariances are:\n', external_drift_kriging_md.in_vars)
# #     print('\nRight hand sides are:\n', external_drift_kriging_md.rhss)
# #     print('\nzks are:\n', external_drift_kriging_md.zk)
# #     print('\nlambdas are:\n', external_drift_kriging_md.lambdas)
# #     print('\nmus_arr is:\n', external_drift_kriging_md.mus_arr)
# #     print('\n\n')
#
# #==============================================================================
# # # indicator kriging
# #==============================================================================
# if ordinary_indicator:
#     print('\a Indicator kriging based on ordinary kriging...')
#     for i in range(n):
#         oindicator_kriging = kriging.OrdinaryIndicatorKriging(xi=xi,
#                                                               yi=yi,
#                                                               zi=zi,
#                                                               xk=xk,
#                                                               yk=yk,
#                                                               lim=lim,
#                                                               model=model)
#         oindicator_kriging.ikrige()
#
#     print('\nDistances are:\n', oindicator_kriging.in_dists)
#     print('\nVariances are:\n', oindicator_kriging.in_vars)
#     print('\nixis are:\n', oindicator_kriging.ixi)
#     print('\niks are:\n', oindicator_kriging.ik)
#     print('\nest_vars are:\n', oindicator_kriging.est_vars)
#     print('\nlambdas are:\n', oindicator_kriging.lambdas)
#     print('\n\n')
#
# if simple_indicator:
#     print('\a Indicator kriging based on simple kriging...')
#     for i in range(n):
#         sindicator_kriging = kriging.SimpleIndicatorKriging(xi=xi,
#                                                             yi=yi,
#                                                             zi=zi,
#                                                             xk=xk,
#                                                             yk=yk,
#                                                             lim=lim,
#                                                             model=model)
#         sindicator_kriging.ikrige()
#
#     print('\nDistances are:\n', sindicator_kriging.in_dists)
#     print('\nVariances are:\n', sindicator_kriging.in_covars)
#     print('\nixis are:\n', sindicator_kriging.ixi)
#     print('\niks are:\n', sindicator_kriging.ik)
#     print('\nest_vars are:\n', sindicator_kriging.est_covars)
#     print('\nlambdas are:\n', sindicator_kriging.lambdas)
#     print('\n\n')

stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' %
      (time.asctime(), stop - start))
