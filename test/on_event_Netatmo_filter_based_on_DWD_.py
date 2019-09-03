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

pyximport.install()

from spinterps import (OrdinaryKriging)

import timeit
import time

import pyproj
import pandas as pd

from pathlib import Path


# =============================================================================

main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
os.chdir(main_dir)

out_plots_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

path_to_data = main_dir / r'oridinary_kriging_compare_DWD_Netatmo/needed_dfs'

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
                         r'edf_ppt_all_dwd_daily_all_stns_combined_.csv')
path_to_netatmo_daily_edf = (path_to_data /
                             r'edf_ppt_all_netatmo_daily_.csv')


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
                          r'edf_ppt_all_dwd_hourly_all_stns_combined_.csv')

path_to_netatmo_hourly_edf = (path_to_data /
                              r'edf_ppt_all_netamo_daily_all_stns_combined_.csv')


# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

path_to_netatmo_coords = path_to_data / r'netatmo_bw_1hour_coords_utm32.csv'

# NETATMO FIRST FILTER
path_to_netatmo_gd_stns = (path_to_data /
                           r'keep_stns_all_neighbor_90_per_60min_.csv')

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

use_dwd_stns_for_kriging = True

normal_kriging = True
qunatile_kriging = False

use_daily_data = True
use_hourly_data = False

use_netatmo_gd_stns = True

use_temporal_filter_after_kriging = True  # run it to filter Netatmo
# =============================================================================
if use_daily_data:
    path_to_netatmo_ppt_data = path_to_netatmo_daily_data  # _temp_filter
    path_to_dwd_ppt_data = path_to_dwd_daily_data
    path_to_netatmo_edf = path_to_netatmo_daily_edf  # _temp_filte
    path_to_dwd_edf = path_to_dwd_daily_edf
    path_netatmo_extremes_df = path_netatmo_daily_extremes_df
    path_dwd_extremes_df = path_dwd_daily_extremes_df

if use_hourly_data:
    path_to_netatmo_ppt_data = path_to_netatmo_hourly_data
    path_to_dwd_ppt_data = path_to_dwd_hourly_data
    path_to_netatmo_edf = path_to_netatmo_hourly_edf
    path_to_dwd_edf = path_to_dwd_hourly_edf
    path_netatmo_extremes_df = path_netatmo_hourly_extremes_df
    path_dwd_extremes_df = path_netatmo_hourly_extremes_df

if normal_kriging:
    netatmo_data_to_use = path_to_netatmo_ppt_data
    dwd_data_to_use = path_to_dwd_ppt_data
    if use_daily_data:
        path_to_dwd_vgs = path_to_dwd_daily_vgs
        out_save_csv = '_daily_ppt_amounts_'
    if use_hourly_data:
        path_to_dwd_vgs = path_to_dwd_hourly_vgs
        out_save_csv = '_hourly_ppt_amounts_'

if qunatile_kriging:
    netatmo_data_to_use = path_to_netatmo_edf
    dwd_data_to_use = path_to_dwd_edf
    if use_daily_data:
        path_to_dwd_vgs = path_to_dwd_daily_edf_vgs
        out_save_csv = '_daily_ppt_edf_'
    if use_hourly_data:
        path_to_dwd_vgs = path_to_dwd_hourly_edf_vgs
        out_save_csv = '_hourly_ppt_edf_'

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
    dwd_in_vals_df.index, format='%Y-%m-%d')

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

df_vgs.index = pd.to_datetime(df_vgs.index, format='%Y-%m-%d')
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
        # check if variogram is 'good'
        if 'Nug' in vgs_model and (
                'Exp' not in vgs_model and 'Sph' not in vgs_model):
            print('**Variogram not valid, looking for alternative\n**',
                  vgs_model)
            try:
                for i in range(4):
                    vgs_model = df_vgs.loc[event_date, str(i)]
                    if type(vgs_model) == np.float:
                        continue
                    if 'Nug' in vgs_model and ('Exp' not in vgs_model or
                                               'Sph' not in vgs_model):
                        continue
                    else:
                        break
            except Exception as msg:
                print(msg)
                print('Only Nugget variogram for this day')
            print('**Changed Variogram model to**\n', vgs_model)

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

        print('\a\a\a Doing Ordinary Kriging \a\a\a')

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
        interpolated_vals = ordinary_kriging.zk
        if use_temporal_filter_after_kriging:

            # calcualte standard deviation of estimated values
            std_est_vals = np.sqrt(ordinary_kriging.est_vars)
            # calculate difference observed and estimated values
            diff_obsv_interp = np.abs(measured_vals - interpolated_vals)

            # use additional temporal filter
            idx_good_stns = np.where(diff_obsv_interp <= 3 * std_est_vals)
            idx_bad_stns = np.where(diff_obsv_interp > 3 * std_est_vals)

            if len(idx_bad_stns[0]) > 0:
                print('Number of Stations with bad index \n',
                      len(idx_bad_stns[0]))
                print('Number of Stations with good index \n',
                      len(idx_good_stns[0]))
                print('**Removing bad stations and saving to new df**')

                # use additional filter
                try:
                    ids_netatmo_stns_gd = np.take(netatmo_df.index,
                                                  idx_good_stns).ravel()
                    ids_netatmo_stns_bad = np.take(netatmo_df.index,
                                                   idx_bad_stns).ravel()

                    df_stns_netatmo_gd_event.loc[
                        event_date,
                        ids_netatmo_stns_bad] = np.nan

                except Exception as msg:
                    print(msg)

    else:
        print('no Variogram for this event')
        continue

if use_temporal_filter_after_kriging:
    df_stns_netatmo_gd_event.to_csv(out_plots_path / (
        r'all_netatmo_%s_temporal_filter.csv'
        % out_save_csv), sep=';',
        float_format='%.2f')

stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s.'
      'Total run time was about %0.4f seconds \a\a\a' %
      (time.asctime(), stop - start))
