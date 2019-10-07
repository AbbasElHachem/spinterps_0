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
import shapefile
import pyproj
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path

from scipy.interpolate import griddata
from scipy.stats import spearmanr as spearmanr
from scipy.stats import pearsonr as pearsonr

plt.ioff()
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.labelsize': 12})

# =============================================================================

main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
os.chdir(main_dir)

out_plots_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

path_to_data = main_dir / r'NetAtmo_BW'

# DAILY DATA
path_netatmo_daily_extremes_df = path_to_data / \
    r'netatmo_daily_maximum_100_days.csv'
path_dwd_daily_extremes_df = path_to_data / r'dwd_daily_maximum_100_days.csv'
path_to_dwd_daily_vgs = path_to_data / r'vg_strs_dwd_daily_ppt_.csv'
path_to_dwd_daily_edf_vgs = path_to_data / r'vg_strs_dwd_daily_edf_.csv'
path_to_dwd_daily_data = (path_to_data /
                          r'all_dwd_daily_ppt_data_combined_2014_2019_.csv')
path_to_netatmo_daily_data = path_to_data / r'all_netatmo_ppt_data_daily_.csv'
path_to_dwd_daily_edf = (path_to_data /
                         r'edf_ppt_all_dwd_daily_.csv')
path_to_netatmo_daily_edf = (path_to_data /
                             r'edf_ppt_all_netatmo_daily_.csv')

# done after filterting based on kriging
path_to_netatmo_daily_data_temp_filter = (
    main_dir / r'oridinary_kriging_compare_DWD_Netatmo' /
    r'all_netatmo__daily_ppt_edf__temporal_filter.csv')
# path_to_netatmo_daily_edf_temp_filte = (
#     path_to_data /
#     r'all_netatmo_edf_data_daily_temporal_filter.csv')


# HOURLY DATA
path_netatmo_hourly_extremes_df = path_to_data / \
    r'netatmo_hourly_maximum_100_hours.csv'
path_dwd_hourly_extremes_df = path_to_data / \
    r'dwd_hourly_maximum_100_hours.csv'
path_to_dwd_hourly_vgs = path_to_data / r'vg_strs_dwd_hourly_ppt_.csv'
path_to_dwd_hourly_edf_vgs = path_to_data / r'vg_strs_dwd_hourly_edf_.csv'

path_to_dwd_hourly_data = (path_to_data /
                           r'all_dwd_hourly_ppt_data_combined_2014_2019_.csv')
path_to_netatmo_hourly_data = path_to_data / \
    r'ppt_all_netatmo_hourly_stns_combined_new.csv'
path_to_dwd_hourly_edf = (path_to_data /
                          r'edf_ppt_all_dwd_hourly_.csv')

path_to_netatmo_hourly_edf = (path_to_data /
                              r'edf_ppt_all_netatmo_hourly_.csv')


# done after filterting based on kriging
path_to_netatmo_hourly_data_temp_filter = (
    path_to_data /
    r'all_netatmo__hourly_ppt_amounts__temporal_filter.csv')
# path_to_netatmo_hourly_edf_temp_filte = (
#     path_to_data /
#     r'all_netatmo__hourly_ppt_edf__temporal_filter.csv')


# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

path_to_netatmo_coords = path_to_data / r'netatmo_bw_1hour_coords_utm32.csv'

# NETATMO FIRST FILTER
path_to_netatmo_gd_stns = (main_dir / r'plots_NetAtmo_ppt_DWD_ppt_correlation_' /
                           r'keep_stns_all_neighbor_90_per_60min_s0.csv')

path_to_shpfile = (
    r"F:\data_from_exchange\Netatmo\Landesgrenze_ETRS89\Landesgrenze_10000_ETRS89_lon_lat.shp")

# =============================================================================
strt_date = '2014-01-01'
end_date = '2019-08-01'

use_netatmo_stns_for_kriging = False
use_dwd_stns_for_kriging = False
use_dwd_and_netatmo_stns_for_kriging = True

normal_kriging = False
qunatile_kriging = True

use_daily_data = True
use_hourly_data = False

# run it to filter Netatmo
use_netatmo_gd_stns = True  # general filter, Indicator kriging
use_temporal_filter_after_kriging = True  # on day filter (in other script)

# =============================================================================
if use_daily_data:
    path_to_netatmo_ppt_data = path_to_netatmo_daily_data  # _temp_filter
    path_to_dwd_ppt_data = path_to_dwd_daily_data
    path_to_netatmo_edf = path_to_netatmo_daily_edf  # _temp_filte
    path_to_dwd_edf = path_to_dwd_daily_edf

    path_netatmo_extremes_df = path_netatmo_daily_extremes_df
    path_dwd_extremes_df = path_dwd_daily_extremes_df
    idx_time_fmt = '%Y-%m-%d'
    plot_label = r'(mm_per_day)'

if use_hourly_data:
    path_to_netatmo_ppt_data = path_to_netatmo_hourly_data
    path_to_dwd_ppt_data = path_to_dwd_hourly_data
    path_to_netatmo_edf = path_to_netatmo_hourly_edf
    path_to_dwd_edf = path_to_dwd_hourly_edf

    path_netatmo_extremes_df = path_netatmo_hourly_extremes_df
    path_dwd_extremes_df = path_netatmo_hourly_extremes_df
    idx_time_fmt = '%Y-%m-%d %H:%M:%S'
    plot_label = r'(mm_per_hour)'

if use_temporal_filter_after_kriging:
    if use_daily_data:
        path_to_netatmo_ppt_data = path_to_netatmo_daily_data_temp_filter
#         path_to_netatmo_edf = path_to_netatmo_daily_edf_temp_filte
    if use_hourly_data:
        path_to_netatmo_ppt_data = path_to_netatmo_hourly_data_temp_filter
#         path_to_netatmo_edf = path_to_netatmo_hourly_edf_temp_filte

if normal_kriging:
    netatmo_data_to_use = path_to_netatmo_ppt_data
    dwd_data_to_use = path_to_dwd_ppt_data
    if use_daily_data:
        path_to_dwd_vgs = path_to_dwd_daily_vgs
    if use_hourly_data:
        path_to_dwd_vgs = path_to_dwd_hourly_vgs
    plot_unit = plot_label
    title_ = r'Amounts'

if qunatile_kriging:
    netatmo_data_to_use = path_to_netatmo_edf
    dwd_data_to_use = path_to_dwd_edf
    if use_daily_data:
        path_to_dwd_vgs = path_to_dwd_daily_edf_vgs
    if use_hourly_data:
        path_to_dwd_vgs = path_to_dwd_hourly_edf_vgs
    plot_unit = 'CDF'
    title_ = r'Quantiles'

if use_temporal_filter_after_kriging:
    title_ = title_ + '_Temporal_filter_used_'
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
    netatmo_in_vals_df.index, format=idx_time_fmt)

netatmo_in_vals_df = netatmo_in_vals_df.loc[strt_date:end_date, :]
netatmo_in_vals_df.dropna(how='all', axis=0, inplace=True)
# daily sums
netatmo_in_vals_df = netatmo_in_vals_df[(0 <= netatmo_in_vals_df) &
                                        (netatmo_in_vals_df <= 300)]

cmn_stns = netatmo_in_coords_df.index.intersection(netatmo_in_vals_df.columns)
netatmo_in_vals_df = netatmo_in_vals_df.loc[:, cmn_stns]
#==============================================================================

if use_netatmo_gd_stns:

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

if use_temporal_filter_after_kriging:
    df_stns_netatmo_gd_event = netatmo_in_vals_df
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
    dwd_in_vals_df.index, format=idx_time_fmt)

dwd_in_vals_df = dwd_in_vals_df.loc[strt_date:end_date, :]
dwd_in_vals_df.dropna(how='all', axis=0, inplace=True)
# daily sums
# dwd_in_vals_df = dwd_in_vals_df[(0 <= dwd_in_vals_df) &
#                                 (dwd_in_vals_df <= 300)]

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
                                     encoding='utf-8',
                                     header=None)

dwd_in_extremes_df = pd.read_csv(path_dwd_extremes_df,
                                 index_col=0,
                                 sep=';',
                                 encoding='utf-8',
                                 header=None)

#==============================================================================
# # VG MODELS
#==============================================================================
df_vgs = pd.read_csv(path_to_dwd_vgs,
                     index_col=0,
                     sep=';',
                     encoding='utf-8')

df_vgs.index = pd.to_datetime(df_vgs.index, format=idx_time_fmt)
df_vgs_models = df_vgs.iloc[:, 0]
df_vgs_models.dropna(how='all', inplace=True)

#==============================================================================
#
#==============================================================================
wgs82 = "+init=EPSG:4326"
utm32 = "+init=EPSG:32632"


def convert_coords_fr_wgs84_to_utm32_(epgs_initial_str, epsg_final_str,
                                      first_coord, second_coord):
    """
    Purpose: Convert points from one reference system to a second
    --------
        In our case the function is used to transform WGS84 to UTM32
        (or vice versa), for transforming the DWD and Netatmo station
        coordinates to same reference system.

        Used for calculating the distance matrix between stations

    Keyword argument:
    -----------------
        epsg_initial_str: EPSG code as string for initial reference system
        epsg_final_str: EPSG code as string for final reference system
        first_coord: numpy array of X or Longitude coordinates
        second_coord: numpy array of Y or Latitude coordinates

    Returns:
    -------
        x, y: two numpy arrays containing the transformed coordinates in 
        the final coordinates system
    """
    initial_epsg = pyproj.Proj(epgs_initial_str)
    final_epsg = pyproj.Proj(epsg_final_str)
    x, y = pyproj.transform(initial_epsg, final_epsg,
                            first_coord, second_coord)
    return x, y


#==============================================================================
#
#==============================================================================
def _plot_interp(
        _interp_x_crds_,
        _interp_y_crds_,
        _interp_z_vals_,
        obs_x_coords,
        obs_y_coords,
        interp_time,
        model,
        interp_type,
        out_figs_dir,
        _nc_vlab,
        _nc_vunits):

    plt.ioff()

    time_str = interp_time

    out_fig_name = f'{interp_type}_{time_str}.png'

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # read and plot shapefile (BW or Germany) should be lon lat

    shp_de = shapefile.Reader(path_to_shpfile)
    xshapefile, yshapefile = [], []
    for shape_ in shp_de.shapeRecords():
        lon = [i[0] for i in shape_.shape.points[:][::-1]]
        lat = [i[1] for i in shape_.shape.points[:][::-1]]
        x0, y0 = convert_coords_fr_wgs84_to_utm32_(
            wgs82, utm32, lon, lat)
        xshapefile.append(x0)
        yshapefile.append(y0)
        ax.scatter(x0, y0, marker='.', c='lightgrey',
                   alpha=0.25, s=1)

    xmesh, ymesh = np.meshgrid(np.linspace(np.min(xshapefile),
                                           np.max(xshapefile), 100),
                               np.linspace(np.min(yshapefile),
                                           np.max(yshapefile), 100))

    zi = griddata((_interp_x_crds_,
                   _interp_y_crds_),
                  _interp_z_vals_,
                  (xmesh, ymesh), method='cubic')

#     # norm = colors.BoundaryNorm(boundaries=np.array([0, 180, 60]), ncolors=256)
    pclr = ax.pcolormesh(xmesh, ymesh, zi,
                         cmap=plt.get_cmap('viridis'),
                         # extend='max', , 1000,origin='lower'
                         vmin=0,
                         vmax=np.int(interpolated_vals.max()) + 1
                         )

    cb = fig.colorbar(pclr)

    cb.set_label(_nc_vlab + ' (' + _nc_vunits + ')')

    ax.scatter(
        obs_x_coords,
        obs_y_coords,
        label='obs. pts.',
        marker='+',
        c='r',
        alpha=0.7)

    ax.legend(framealpha=0.5)

    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    # ax.set_xticks([xmesh[0]])
    # ax.set_yticks([ymesh[0]])
    title = (f'Time: {time_str}\n(VG: {model})\n')
#         f'Min.: {grd_min:0.4f}, Max.: {grd_max:0.4f}')

    ax.set_title(title)

    #plt.setp(ax.get_xmajorticklabels(), rotation=70)
    #ax.set_aspect('equal', 'datalim')

    plt.savefig(str(out_figs_dir / out_fig_name), bbox_inches='tight')
    plt.close()
    return


print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
start = timeit.default_timer()  # to get the runtime of the program

#==============================================================================
# # ordinary kriging
#==============================================================================


for event_date in df_vgs_models.index:
    # if event_date == '2018-05-14':
    #   raise Exception
    if ((event_date in df_vgs_models.index) and
        (event_date in dwd_in_vals_df.index) and
            (event_date in netatmo_in_vals_df.index)):
        print('Event date is', event_date)

        vgs_model = df_vgs_models.loc[event_date]
        # check if variogram is 'good'
        if 'Nug' in vgs_model and (
                'Exp' not in vgs_model and 'Sph' not in vgs_model):
            print('**Variogram %s not valid,\n looking for alternative\n**'
                  % vgs_model)
            try:
                for i in range(1, 4):
                    vgs_model = df_vgs.loc[event_date, str(i)]
                    if type(vgs_model) == np.float:
                        continue
                    if 'Nug' in vgs_model and ('Exp' not in vgs_model and
                                               'Sph' not in vgs_model):
                        continue
                    else:
                        break

            except Exception as msg:
                print(msg)
                print('Only Nugget variogram for this day')
        if type(vgs_model) != np.float:
            print('**Changed Variogram model to**\n', vgs_model)

            # DWD data and coords
            dwd_df = dwd_in_vals_df.loc[event_date, :].dropna(how='all')
            dwd_vals = dwd_df.values
            dwd_coords = dwd_in_coords_df.loc[dwd_df.index]
            x_dwd, y_dwd = dwd_coords.X.values, dwd_coords.Y.values

            # Netatmo data and coords
            netatmo_df = netatmo_in_vals_df.loc[event_date, :].dropna(
                how='all')
            netatmo_vals = netatmo_df.values
            netatmo_coords = netatmo_in_coords_df.loc[netatmo_df.index]
            x_netatmo, y_netatmo = netatmo_coords.X.values, netatmo_coords.Y.values

            # TODO: add filter on minimum number of stations
            print('\a\a\a Doing Ordinary Kriging \a\a\a')

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

            if use_dwd_and_netatmo_stns_for_kriging:
                print('using DWD and Netatmo stations to find Netatmo values')

                dwd_netatmo_xcoords = np.concatenate(
                    [x_dwd, x_netatmo])
                dwd_netatmo_ycoords = np.concatenate(
                    [y_dwd, y_netatmo])

                coords = np.array([(x, y) for x, y in zip(dwd_netatmo_xcoords,
                                                          dwd_netatmo_ycoords)])
                #dwd_netatmo_ppt = np.hstack((ppt_dwd_vals, ppt_netatmo_vals))
                dwd_netatmo_ppt = np.concatenate([dwd_vals,
                                                  netatmo_vals])

                measured_vals = dwd_vals
                used_vals = dwd_netatmo_ppt

                xlabel = 'Netatmo and DWD observed values'
                ylabel = 'Netatmo and DWD interpolated values using DWD data'
                measured_stns = 'DWD'
                used_stns = 'DWD and Netatmo'
                plot_title_acc = '_using_DWD_and_NEtatmo_stations_to_find_DWD_values_'

                ordinary_kriging = OrdinaryKriging(
                    xi=dwd_netatmo_xcoords,
                    yi=dwd_netatmo_ycoords,
                    zi=dwd_netatmo_ppt,
                    xk=x_dwd,
                    yk=y_dwd,
                    model=vgs_model)
            try:
                ordinary_kriging.krige()
            except Exception as msg:
                print('Error while Kriging', msg)
                continue

            # print('\nDistances are:\n', ordinary_kriging.in_dists)
            # print('\nVariances are:\n', ordinary_kriging.in_vars)
            # print('\nRight hand sides are:\n', ordinary_kriging.rhss)
            # print('\nzks are:', ordinary_kriging.zk)
            # print('\nest_vars are:\n', ordinary_kriging.est_vars)
            # print('\nlambdas are:\n', ordinary_kriging.lambdas)
            # print('\nmus are:\n', ordinary_kriging.mus)
            # print('\n\n')

            # interpolated vals
            interpolated_vals = ordinary_kriging.zk.copy()
            if qunatile_kriging and np.max(interpolated_vals) > 1:
                interpolated_vals[interpolated_vals > 1] = 1
                # raise Exception
            event_date = str(event_date).replace('-', '_').replace(':', '_')

            # calculate correlation between observed and interpolated
            pear_corr = pearsonr(measured_vals, interpolated_vals)[0]
            spr_corr = spearmanr(measured_vals, interpolated_vals)[0]

            #==================================================================
            # PLOT SCATTER INTERPOLATED VS OBSERVED
            #==================================================================
            plt.ioff()
            fig, ax = plt.subplots(figsize=(16, 12), dpi=100)

            ax.scatter(measured_vals, interpolated_vals,
                       color='r', marker='*', alpha=0.75, s=25,
                       label=(('Nbr used %s stns %d \n'
                               'Nbr used %s stns %d')
                              % (measured_stns, measured_vals.shape[0],
                                 used_stns, used_vals.shape[0])))

            ax.set_xlim(
                [-0.1, max(measured_vals.max(),
                           interpolated_vals.max()) + .2])
            ax.set_ylim(
                [-0.1, max(measured_vals.max(),
                           interpolated_vals.max()) + .2])

            ax.plot([0, max(measured_vals.max(), interpolated_vals.max()) + .2],
                    [0, max(measured_vals.max(), interpolated_vals.max()) + .2],
                    color='k', alpha=0.25, linestyle='--')

            ax.set_title(('Observed vs Interpolated Rainfall %s \n %s \n '
                          '%s \n'
                          'Event Date:%s and Variogram model: %s\n'
                          'Pearson Corr: %0.2f and Spearman Corr: %0.2f'
                          % (title_, plot_unit, plot_title_acc,
                             event_date, vgs_model, pear_corr, spr_corr)))

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.legend(loc=0)
            ax.grid(alpha=0.5)
            try:
                plt.savefig(out_plots_path /
                            (r"observed_vs_interpolated_%s_%s_%s_%s.png"
                             % (title_, plot_unit, event_date, plot_title_acc)),
                            frameon=True, papertype='a4',
                            bbox_inches='tight', pad_inches=.2)
            except Exception as msg:
                print('Error while saving', msg)
                continue
            plt.clf()
            plt.close('all')

    #         _plot_interp(
    #             x_dwd,
    #             y_dwd,
    #             interpolated_vals,
    #             x_netatmo,
    #             y_netatmo,
    #             event_date,
    #             vgs_model,
    #             'OK',
    #             out_plots_path,
    #             plot_unit,
    #             '')
        else:
            print('No suitable variogram was found for this event, skipping')
            continue
    else:
        print('no Variogram for this event')
        continue


stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' %
      (time.asctime(), stop - start))

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
