# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

TODO: Add doc
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

import pandas as pd

from pathlib import Path


# =============================================================================

main_dir = Path(r'X:\hiwi\ElHachem\Prof_Bardossy\Extremes')
os.chdir(main_dir)

out_plots_path = main_dir / r'oridinary_kriging_compare_DWD_Netatmo'

path_to_data = main_dir / r'NetAtmo_BW'

path_to_vgs = main_dir / r'kriging_ppt_netatmo'

# COORDINATES
path_to_dwd_coords = (path_to_data /
                      r'station_coordinates_names_hourly_only_in_BW_utm32.csv')

path_to_netatmo_coords = path_to_data / r'netatmo_bw_1hour_coords_utm32.csv'

# NETATMO FIRST FILTER
path_to_netatmo_gd_stns = (main_dir / r'plots_NetAtmo_ppt_DWD_ppt_correlation_' /
                           r'keep_stns_all_neighbor_98_per_60min_s0.csv')

#==============================================================================
#
#==============================================================================

use_dwd_stns_for_kriging = True

strt_date = '2015-01-01'
end_date = '2019-09-01'

resample_frequencies = ['60min', '120min', '180min',
                        '360min', '720min', '1440min']
#==============================================================================
#
#==============================================================================


# Netatmo Coords
netatmo_in_coords_df = pd.read_csv(path_to_netatmo_coords,
                                   index_col=0,
                                   sep=';',
                                   encoding='utf-8')

# DWD Coords

dwd_in_coords_df = pd.read_csv(path_to_dwd_coords,
                               index_col=0,
                               sep=';',
                               encoding='utf-8')
# added by Abbas, for DWD stations
stndwd_ix = ['0' * (5 - len(str(stn_id))) + str(stn_id)
             if len(str(stn_id)) < 5 else str(stn_id)
             for stn_id in dwd_in_coords_df.index]

dwd_in_coords_df.index = stndwd_ix
dwd_in_coords_df.index = list(map(str, dwd_in_coords_df.index))


# Netatmo first filter
df_gd_stns = pd.read_csv(path_to_netatmo_gd_stns,
                         index_col=0,
                         sep=';',
                         encoding='utf-8')


#==============================================================================
#
#==============================================================================
for temp_agg in resample_frequencies:

    # path to data
    #=========================================================================
    out_save_csv = '_%s_ppt_edf_' % temp_agg

    path_to_dwd_edf = (path_to_data /
                       (r'edf_ppt_all_dwd_%s_.csv' % temp_agg))

    path_to_netatmo_edf = (path_to_data /
                           (r'edf_ppt_all_netatmo_good_stns_%s_.csv' % temp_agg))

    path_to_dwd_vgs = path_to_vgs / \
        (r'vg_strs_dwd_%s_maximum_100_event.csv' % temp_agg)

    path_dwd_extremes_df = path_to_data / \
        (r'dwd_%s_maximum_100_event.csv' % temp_agg)

    # DWD DATA
    #=========================================================================
    dwd_in_vals_df = pd.read_csv(
        path_to_dwd_edf, sep=';', index_col=0, encoding='utf-8')

    dwd_in_vals_df.index = pd.to_datetime(
        dwd_in_vals_df.index, format='%Y-%m-%d')

    dwd_in_vals_df = dwd_in_vals_df.loc[strt_date:end_date, :]
    dwd_in_vals_df.dropna(how='all', axis=0, inplace=True)

    # NETAMO DATA
    #=========================================================================
    netatmo_in_vals_df = pd.read_csv(
        path_to_netatmo_edf, sep=';',
        index_col=0,
        encoding='utf-8', engine='c')

    netatmo_in_vals_df.index = pd.to_datetime(
        netatmo_in_vals_df.index, format='%Y-%m-%d')

    netatmo_in_vals_df = netatmo_in_vals_df.loc[strt_date:end_date, :]
    netatmo_in_vals_df.dropna(how='all', axis=0, inplace=True)
    # apply first filter
    good_netatmo_stns = df_gd_stns.loc[:, 'Stations'].values.ravel()

    in_vals_df = netatmo_in_vals_df.loc[:, good_netatmo_stns]

    netatmo_in_coords_df = netatmo_in_coords_df.loc[good_netatmo_stns, :]
    cmn_stns = netatmo_in_coords_df.index.intersection(
        netatmo_in_vals_df.columns)

    netatmo_in_vals_df = netatmo_in_vals_df.loc[:, cmn_stns]
    # DWD Extremes
    #=========================================================================
    dwd_in_extremes_df = pd.read_csv(path_dwd_extremes_df,
                                     index_col=0,
                                     sep=';',
                                     parse_dates=True,
                                     infer_datetime_format=True,
                                     encoding='utf-8',
                                     header=None)
    # # VG MODELS
    #==========================================================================
    df_vgs = pd.read_csv(path_to_dwd_vgs,
                         index_col=0,
                         sep=';',
                         encoding='utf-8')

    df_vgs.index = pd.to_datetime(df_vgs.index, format='%Y-%m-%d')
    df_vgs_models = df_vgs.iloc[:, 0]
    df_vgs_models.dropna(how='all', inplace=True)

    # DF to hold results
    #==========================================================================

    df_stns_netatmo_gd_event = pd.DataFrame(
        index=df_vgs_models.index,
        columns=netatmo_in_vals_df.columns,
        data=np.ones(
            shape=(df_vgs_models.index.shape[0],
                   netatmo_in_vals_df.columns.shape[0])))

    # ordinary kriging
    #==========================================================================
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    start = timeit.default_timer()  # to get the runtime of the program
    #=========================================================================

    for event_date in df_vgs_models.index:

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
            netatmo_df = netatmo_in_vals_df.loc[event_date, :].dropna(
                how='all')
            netatmo_vals = netatmo_df.values
            netatmo_coords = netatmo_in_coords_df.loc[netatmo_df.index]
            x_netatmo, y_netatmo = netatmo_coords.X.values, netatmo_coords.Y.values

            # Combine both coordinates and data
            x_dwd_netatmo_comb = np.concatenate((x_dwd, x_netatmo))
            y_dwd_netatmo_comb = np.concatenate((y_dwd, y_netatmo))
            ppt_dwd_netatmo_comb = np.concatenate((dwd_vals, netatmo_vals))

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

            # calcualte standard deviation of estimated values
            std_est_vals = np.sqrt(ordinary_kriging.est_vars)
            # calculate difference observed and estimated values
            diff_obsv_interp = np.abs(measured_vals - interpolated_vals)

            #==================================================================
            # # use additional temporal filter
            #==================================================================
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
                        ids_netatmo_stns_bad] = -999

                except Exception as msg:
                    print(msg)

        else:
            print('no Variogram for this event')
            continue
    out_save_csv = out_save_csv + plot_title_acc

    df_stns_netatmo_gd_event.to_csv(out_plots_path / (
        r'all_netatmo_%s_temporal_filter_98perc_.csv'
        % out_save_csv), sep=';',
        float_format='%.0f')

    stop = timeit.default_timer()  # Ending time
    print('\n\a\a\a Done with everything on %s.'
          'Total run time was about %0.4f seconds \a\a\a' %
          (time.asctime(), stop - start))
